"""Binance REST API provider implementation.

This module implements the BaseProvider interface for Binance's cryptocurrency
exchange, providing access to spot market instruments and OHLCV candle data.

Binance API Documentation: https://binance-docs.github.io/apidocs/spot/en/

Features:
- Cryptocurrency instrument discovery
- OHLCV candle data (klines)
- Support for Binance.com and Binance.us
- Automatic pagination for large date ranges
"""

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    ValidationError,
)
from liq.data.providers.base import BaseProvider


class BinanceProvider(BaseProvider):
    """Binance REST API provider for cryptocurrency data.

    Provides access to Binance's spot market instruments and historical
    candle data through their REST API.

    Example:
        provider = BinanceProvider()  # Public API, no auth needed for market data

        # Fetch instruments
        instruments = provider.list_instruments()

        # Fetch OHLCV data
        bars = provider.fetch_bars(
            "BTC_USDT",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            timeframe="1h"
        )
    """

    # Binance API base URLs
    BASE_URL = "https://api.binance.com"
    US_BASE_URL = "https://api.binance.us"

    # Timeframe to Binance interval mapping
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
    }

    # Timeframe to minutes mapping for pagination
    TIMEFRAME_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
    }

    # Maximum candles per Binance API request
    MAX_CANDLES_PER_REQUEST = 1000

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        use_us: bool = False,
        timeout: float = 30.0,
    ) -> None:
        """Initialize Binance provider.

        Args:
            api_key: Optional API key (not needed for public endpoints)
            api_secret: Optional API secret
            use_us: Use Binance.us instead of Binance.com
            timeout: Request timeout in seconds (default: 30)
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._use_us = use_us
        self._timeout = timeout

        self._base_url = self.US_BASE_URL if use_us else self.BASE_URL

        # Client created lazily
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["X-MBX-APIKEY"] = self._api_key
            self._client = httpx.Client(headers=headers, timeout=self._timeout)
        return self._client

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "binance_us" if self._use_us else "binance"

    @property
    def supported_timeframes(self) -> list[str]:
        """Return supported timeframes for Binance."""
        return list(self.TIMEFRAME_MAP.keys())

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch OHLCV candle data from Binance with automatic pagination.

        Args:
            symbol: Symbol in canonical format (e.g., "BTC_USDT")
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ProviderError: If API call fails
            RateLimitError: If rate limit is exceeded
        """
        self.validate_timeframe(timeframe)

        interval = self.TIMEFRAME_MAP[timeframe]
        minutes_per_candle = self.TIMEFRAME_MINUTES[timeframe]

        # Convert canonical symbol to Binance format (BTC_USDT -> BTCUSDT)
        binance_symbol = symbol.replace("_", "")

        # Convert dates to datetimes
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)

        # Calculate chunk size
        chunk_minutes = self.MAX_CANDLES_PER_REQUEST * minutes_per_candle
        chunk_delta = timedelta(minutes=chunk_minutes)

        all_bars: list[dict[str, Any]] = []
        current_start = start_dt

        try:
            while current_start < end_dt:
                current_end = min(current_start + chunk_delta, end_dt)
                current_start_ms = int(current_start.timestamp() * 1000)
                current_end_ms = int(current_end.timestamp() * 1000)

                chunk_bars = self._fetch_klines(
                    binance_symbol, interval, current_start_ms, current_end_ms
                )
                all_bars.extend(chunk_bars)

                # Move to next chunk
                if chunk_bars:
                    last_timestamp = chunk_bars[-1]["timestamp"]
                    current_start = last_timestamp + timedelta(minutes=minutes_per_candle)
                else:
                    current_start = current_end

            return self.bars_to_dataframe(all_bars)

        except (RateLimitError, ProviderError, ValidationError):
            raise
        except (httpx.RequestError, KeyError, ValueError, TypeError) as e:
            raise ProviderError(f"Failed to fetch Binance candles: {e}") from e

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[dict[str, Any]]:
        """Fetch a single chunk of kline data."""
        url = f"{self._base_url}/api/v3/klines"
        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": self.MAX_CANDLES_PER_REQUEST,
        }

        response = self._get_client().get(url, params=params)

        if response.status_code == 429:
            raise RateLimitError("Binance API rate limit exceeded")
        if response.status_code == 401:
            raise AuthenticationError("Invalid Binance API credentials")
        if response.status_code != 200:
            raise ProviderError(f"Binance API error: {response.status_code} - {response.text}")

        klines = response.json()
        bars: list[dict[str, Any]] = []

        for k in klines:
            # Binance kline format:
            # [0] open time, [1] open, [2] high, [3] low, [4] close, [5] volume, ...
            if len(k) < 6:
                raise ValidationError(f"Malformed kline: expected 6+ fields, got {len(k)}")

            if k[0] is None:
                raise ValidationError("Missing timestamp in kline data")

            timestamp = datetime.fromtimestamp(k[0] / 1000, tz=UTC)

            for i, field in enumerate(["open", "high", "low", "close", "volume"], 1):
                if k[i] is None:
                    raise ValidationError(f"Missing {field} at {timestamp.isoformat()}")

            bars.append({
                "timestamp": timestamp,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        return bars

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        """List available cryptocurrency instruments from Binance.

        Args:
            asset_class: Must be "crypto" or None

        Returns:
            DataFrame with instrument metadata

        Raises:
            ProviderError: If asset_class is not "crypto"
        """
        if asset_class is not None and asset_class != "crypto":
            raise ProviderError(f"Binance only supports crypto, got: {asset_class}")

        try:
            url = f"{self._base_url}/api/v3/exchangeInfo"
            response = self._get_client().get(url)

            if response.status_code == 429:
                raise RateLimitError("Binance API rate limit exceeded")
            if response.status_code != 200:
                raise ProviderError(f"Binance API error: {response.status_code}")

            data = response.json()
            instruments = []

            for symbol_info in data.get("symbols", []):
                if symbol_info.get("status") != "TRADING":
                    continue

                base = symbol_info.get("baseAsset", "")
                quote = symbol_info.get("quoteAsset", "")
                canonical_symbol = f"{base}_{quote}"

                instruments.append({
                    "symbol": canonical_symbol,
                    "name": f"{base}/{quote}",
                    "asset_class": "crypto",
                    "base_currency": base,
                    "quote_currency": quote,
                    "status": symbol_info.get("status", ""),
                })

            return pl.DataFrame(instruments)

        except RateLimitError:
            raise
        except (httpx.RequestError, KeyError, ValueError) as e:
            raise ProviderError(f"Failed to fetch Binance instruments: {e}") from e

    def validate_credentials(self) -> bool:
        """Validate Binance API connectivity.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            url = f"{self._base_url}/api/v3/exchangeInfo"
            response = self._get_client().get(url)
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()

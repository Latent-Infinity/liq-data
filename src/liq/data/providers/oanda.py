"""OANDA v20 API provider implementation.

This module implements the BaseProvider interface for OANDA's forex trading platform,
providing access to forex instruments and OHLCV candle data.

OANDA API Documentation: https://developer.oanda.com/rest-live-v20/introduction/

Features:
- Forex instrument discovery
- OHLCV candle data with bid/ask spreads
- Automatic pagination for large date ranges (exceeding 5000 candles)
- Practice and live environment support
- Automatic timeframe mapping to OANDA granularities
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


class OandaProvider(BaseProvider):
    """OANDA v20 API provider for forex data.

    Provides access to OANDA's forex instruments and historical candle data
    through their v20 REST API.

    Example:
        provider = OandaProvider(
            api_key="your_api_key",
            account_id="your_account_id",
            environment="practice"  # or "live"
        )

        # Fetch instruments
        instruments = provider.list_instruments()

        # Fetch OHLCV data
        bars = provider.fetch_bars(
            "EUR_USD",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            timeframe="1h"
        )
    """

    # OANDA API base URLs
    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    # Timeframe to OANDA granularity mapping
    TIMEFRAME_MAP = {
        "1m": "M1",
        "5m": "M5",
        "15m": "M15",
        "30m": "M30",
        "1h": "H1",
        "4h": "H4",
        "1d": "D",
        "1w": "W",
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

    # Maximum candles per OANDA API request
    MAX_CANDLES_PER_REQUEST = 5000

    def __init__(
        self,
        api_key: str,
        account_id: str,
        environment: str = "practice",
        timeout: float = 30.0,
    ) -> None:
        """Initialize OANDA provider with credentials.

        Args:
            api_key: OANDA API authentication token
            account_id: OANDA account ID
            environment: "practice" for demo or "live" for real trading
            timeout: Request timeout in seconds (default: 30)
        """
        self._api_key = api_key
        self._account_id = account_id
        self._environment = environment
        self._timeout = timeout

        # Set base URL based on environment
        self._base_url = self.PRACTICE_URL if environment == "practice" else self.LIVE_URL

        # Client created lazily
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "oanda"

    @property
    def supported_timeframes(self) -> list[str]:
        """Return supported timeframes for OANDA."""
        return list(self.TIMEFRAME_MAP.keys())

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch OHLCV candle data from OANDA with automatic pagination.

        Args:
            symbol: OANDA instrument name (e.g., "EUR_USD")
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ProviderError: If timeframe is unsupported or API call fails
            AuthenticationError: If API credentials are invalid
            RateLimitError: If rate limit is exceeded
        """
        self.validate_timeframe(timeframe)

        granularity = self.TIMEFRAME_MAP[timeframe]
        minutes_per_candle = self.TIMEFRAME_MINUTES[timeframe]

        # Convert dates to datetimes
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)

        # Calculate chunk size (5000 candles worth)
        chunk_minutes = self.MAX_CANDLES_PER_REQUEST * minutes_per_candle
        chunk_delta = timedelta(minutes=chunk_minutes)

        all_bars: list[dict[str, Any]] = []
        current_start = start_dt

        try:
            # Paginate through the date range
            while current_start < end_dt:
                current_end = min(current_start + chunk_delta, end_dt)

                chunk_bars = self._fetch_bars_chunk(
                    symbol, current_start, current_end, granularity
                )
                all_bars.extend(chunk_bars)

                # Move to next chunk
                if chunk_bars:
                    last_timestamp = chunk_bars[-1]["timestamp"]
                    current_start = last_timestamp + timedelta(minutes=minutes_per_candle)
                else:
                    current_start = current_end

            return self.bars_to_dataframe(all_bars)

        except (AuthenticationError, RateLimitError, ProviderError, ValidationError):
            raise
        except (httpx.RequestError, KeyError, ValueError, TypeError) as e:
            raise ProviderError(f"Failed to fetch OANDA candles: {e}") from e

    def _fetch_bars_chunk(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        granularity: str,
    ) -> list[dict[str, Any]]:
        """Fetch a single chunk of OHLCV data (up to 5000 candles)."""
        from_time = start_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        to_time = end_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        url = (
            f"{self._base_url}/v3/instruments/{symbol}/candles"
            f"?from={from_time}&to={to_time}&granularity={granularity}"
            f"&price=M"  # Request mid prices
        )

        response = self._get_client().get(url)

        if response.status_code == 401:
            raise AuthenticationError("Invalid OANDA API credentials")
        if response.status_code == 429:
            raise RateLimitError("OANDA API rate limit exceeded")
        if response.status_code != 200:
            raise ProviderError(f"OANDA API error: {response.status_code} - {response.text}")

        data = response.json()
        bars: list[dict[str, Any]] = []

        for candle in data.get("candles", []):
            # Skip incomplete candles
            if not candle.get("complete", False):
                continue

            if "time" not in candle:
                raise ValidationError("Missing timestamp in candle data")

            # Parse timestamp (OANDA uses RFC3339 format)
            timestamp_str = candle["time"]
            timestamp = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00").split(".")[0] + "+00:00"
            )

            if "mid" not in candle:
                raise ValidationError(f"Missing mid price data at {timestamp_str}")

            mid = candle["mid"]
            required = ["o", "h", "l", "c"]
            for field in required:
                if field not in mid:
                    raise ValidationError(f"Missing OHLC field '{field}' at {timestamp_str}")

            bars.append({
                "timestamp": timestamp,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": float(candle.get("volume", 0)),
            })

        return bars

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        """List available forex instruments from OANDA.

        Args:
            asset_class: Must be "forex" or None (OANDA only supports forex)

        Returns:
            DataFrame with instrument metadata

        Raises:
            ProviderError: If asset_class is not "forex"
        """
        if asset_class is not None and asset_class != "forex":
            raise ProviderError(f"OANDA only supports forex, got: {asset_class}")

        try:
            url = f"{self._base_url}/v3/accounts/{self._account_id}/instruments"
            response = self._get_client().get(url)

            if response.status_code == 401:
                raise AuthenticationError("Invalid OANDA API credentials")
            if response.status_code == 429:
                raise RateLimitError("OANDA API rate limit exceeded")
            if response.status_code != 200:
                raise ProviderError(f"OANDA API error: {response.status_code}")

            data = response.json()
            instruments = []

            for inst in data.get("instruments", []):
                instruments.append({
                    "symbol": inst["name"],
                    "name": inst.get("displayName", inst["name"]),
                    "asset_class": "forex",
                    "type": inst.get("type", "CURRENCY"),
                })

            return pl.DataFrame(instruments)

        except (AuthenticationError, RateLimitError):
            raise
        except (httpx.RequestError, KeyError, ValueError) as e:
            raise ProviderError(f"Failed to fetch OANDA instruments: {e}") from e

    def validate_credentials(self) -> bool:
        """Validate OANDA API credentials.

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            url = f"{self._base_url}/v3/accounts/{self._account_id}"
            response = self._get_client().get(url)
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def __del__(self) -> None:
        """Clean up HTTP client."""
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()

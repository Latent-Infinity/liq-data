"""Coinbase Exchange API provider implementation.

This module implements the BaseProvider interface for Coinbase Exchange API,
providing access to cryptocurrency market data.

Coinbase Exchange API Documentation: https://docs.cdp.coinbase.com/exchange/docs/welcome

Features:
- Cryptocurrency pair data (BTC-USD, ETH-USD, etc.)
- OHLCV bar data via candles endpoint
- HMAC-SHA256 authentication
- Rate limiting handling
- Automatic pagination for large date ranges (max 300 candles per request)
"""

import base64
import hashlib
import hmac
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from liq.data.providers.base import BaseProvider


class CoinbaseProvider(BaseProvider):
    """Coinbase Exchange API provider for cryptocurrency data.

    Provides access to Coinbase Exchange market data through their REST API
    using HMAC-SHA256 authentication.

    Example:
        provider = CoinbaseProvider(
            api_key="your_api_key",
            api_secret="your_api_secret",
            passphrase="your_passphrase",
        )

        # Fetch crypto data
        bars = provider.fetch_bars(
            "BTC-USD",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            timeframe="1h"
        )
    """

    # Coinbase API URLs
    BASE_URL = "https://api.exchange.coinbase.com"

    # Timeframe to Coinbase granularity mapping (in seconds)
    TIMEFRAME_MAP = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "6h": 21600,
        "1d": 86400,
    }

    # Maximum candles per request (Coinbase limit is 300)
    MAX_CANDLES_PER_REQUEST = 300

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        passphrase: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize Coinbase provider.

        Args:
            api_key: Coinbase Exchange API key (optional for public data)
            api_secret: Coinbase Exchange API secret (optional for public data)
            passphrase: Coinbase Exchange API passphrase (optional for public data)
            timeout: Request timeout in seconds (default: 30)
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._timeout = timeout
        self._auth_enabled = bool(api_key and api_secret and passphrase)

        # HTTP client created lazily
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    @property
    def name(self) -> str:
        """Provider name identifier."""
        return "coinbase"

    @property
    def supported_timeframes(self) -> list[str]:
        """Return supported timeframes for Coinbase."""
        return list(self.TIMEFRAME_MAP.keys())

    def _generate_signature(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = "",
    ) -> str:
        """Generate HMAC-SHA256 signature for API authentication.

        Args:
            timestamp: Unix timestamp as string
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body (empty for GET)

        Returns:
            Base64-encoded signature
        """
        message = f"{timestamp}{method}{request_path}{body}"
        assert self._api_secret is not None  # Validated in _auth_enabled check
        hmac_key = base64.b64decode(self._api_secret)
        signature = hmac.new(
            hmac_key,
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(signature.digest()).decode("utf-8")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            ProviderError: If API call fails
        """
        client = self._get_client()

        headers: dict[str, str] | None = None

        url = f"{self.BASE_URL}{endpoint}"

        # Authenticated path if creds provided; otherwise public GET
        if self._auth_enabled:
            request_path = endpoint
            if params:
                query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
                request_path = f"{endpoint}?{query_string}"

            timestamp = str(int(time.time()))
            signature = self._generate_signature(timestamp, method, request_path)

            headers = {
                "CB-ACCESS-KEY": self._api_key or "",
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-PASSPHRASE": self._passphrase or "",
                "Content-Type": "application/json",
            }

        try:
            response = client.request(method, url, params=params, headers=headers)
        except httpx.RequestError as e:
            raise ProviderError(f"Coinbase API request failed: {e}") from e

        if response.status_code == 401:
            raise AuthenticationError(
                f"Coinbase authentication failed: {response.text}"
            )

        if response.status_code == 429:
            raise RateLimitError("Coinbase rate limit exceeded")

        if response.status_code != 200:
            raise ProviderError(
                f"Coinbase API error: {response.status_code} - {response.text}"
            )

        return response.json()

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Coinbase API.

        Converts underscore format to hyphen format and uppercases.

        Args:
            symbol: Input symbol (e.g., "btc_usd" or "BTC-USD")

        Returns:
            Normalized symbol for API calls (e.g., "BTC-USD")
        """
        return symbol.upper().replace("_", "-")

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch OHLCV bar data from Coinbase with automatic pagination.

        Args:
            symbol: Trading pair (e.g., "BTC-USD" or "BTC_USD")
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Bar timeframe (default: "1d")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ProviderError: If API call fails or timeframe unsupported
            RateLimitError: If rate limit is exceeded
        """
        self.validate_timeframe(timeframe)

        granularity = self.TIMEFRAME_MAP[timeframe]
        api_symbol = self._normalize_symbol(symbol)

        all_bars: list[dict[str, Any]] = []
        start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=UTC)
        end_dt = datetime.combine(end, datetime.max.time()).replace(tzinfo=UTC)

        current_start = start_dt

        while current_start < end_dt:
            # Calculate end of this chunk (max 300 candles)
            chunk_seconds = granularity * self.MAX_CANDLES_PER_REQUEST
            current_end = min(
                current_start + timedelta(seconds=chunk_seconds),
                end_dt,
            )

            params: dict[str, Any] = {
                "granularity": str(granularity),
                "start": current_start.isoformat(),
                "end": current_end.isoformat(),
            }

            data = self._make_request(
                "GET", f"/products/{api_symbol}/candles", params
            )

            if not data:
                break

            # Coinbase returns [timestamp, low, high, open, close, volume]
            for candle in data:
                ts = datetime.fromtimestamp(candle[0], tz=UTC)
                if ts < start_dt or ts > end_dt:
                    continue

                all_bars.append({
                    "timestamp": ts,
                    "open": float(candle[3]),
                    "high": float(candle[2]),
                    "low": float(candle[1]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })

            # Move to next chunk
            current_start = current_end

            # If no data returned, stop
            if len(data) == 0:
                break

        return self.bars_to_dataframe(all_bars)

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:  # noqa: ARG002
        """List available trading pairs from Coinbase.

        Args:
            asset_class: Not used for Coinbase (all are crypto)

        Returns:
            DataFrame with instrument metadata
        """
        data = self._make_request("GET", "/products")

        all_products: list[dict[str, Any]] = []

        for product in data:
            # Filter out disabled trading pairs
            if product.get("trading_disabled", False):
                continue
            if product.get("status") != "online":
                continue

            all_products.append({
                "symbol": product.get("id", ""),
                "name": product.get("display_name", ""),
                "asset_class": "crypto",
                "exchange": "coinbase",
                "type": "spot",
                "base_currency": product.get("base_currency", ""),
                "quote_currency": product.get("quote_currency", ""),
            })

        if not all_products:
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "name": pl.Utf8,
                    "asset_class": pl.Utf8,
                    "exchange": pl.Utf8,
                    "type": pl.Utf8,
                    "base_currency": pl.Utf8,
                    "quote_currency": pl.Utf8,
                }
            )

        return pl.DataFrame(all_products)

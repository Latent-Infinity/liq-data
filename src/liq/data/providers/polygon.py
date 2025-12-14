"""Polygon.io API provider implementation.

This module implements the BaseProvider interface for Polygon.io API,
providing access to US equities, options, forex, and crypto market data.

Polygon API Documentation: https://polygon.io/docs

Features:
- US stocks data (NYSE, NASDAQ, etc.)
- OHLCV bar data via aggregates endpoint
- Bearer token authentication
- Rate limiting handling (5 req/min free tier)
- Automatic pagination for large datasets
"""

from datetime import UTC, date, datetime
from typing import Any

import httpx
import polars as pl

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from liq.data.providers.base import BaseProvider


class PolygonProvider(BaseProvider):
    """Polygon.io API provider for market data.

    Provides access to US equities and other market data through Polygon's REST API
    using Bearer token authentication.

    Example:
        provider = PolygonProvider(api_key="your_api_key")

        # Fetch stock data
        bars = provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            timeframe="1h"
        )
    """

    # Polygon API URL
    BASE_URL = "https://api.polygon.io"

    # Timeframe to Polygon timespan/multiplier mapping
    TIMEFRAME_MAP = {
        "1m": ("minute", 1),
        "5m": ("minute", 5),
        "15m": ("minute", 15),
        "30m": ("minute", 30),
        "1h": ("hour", 1),
        "4h": ("hour", 4),
        "1d": ("day", 1),
        "1w": ("week", 1),
    }

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize Polygon provider.

        Args:
            api_key: Polygon API key (required)
            timeout: Request timeout in seconds (default: 30)

        Raises:
            ValueError: If api_key is not provided
        """
        if not api_key:
            raise ValueError("api_key is required for Polygon provider")

        self._api_key = api_key
        self._timeout = timeout

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
        return "polygon"

    @property
    def supported_timeframes(self) -> list[str]:
        """Return supported timeframes for Polygon."""
        return list(self.TIMEFRAME_MAP.keys())

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

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = client.request(method, url, params=params, headers=headers)
        except httpx.RequestError as e:
            raise ProviderError(f"Polygon API request failed: {e}") from e

        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"Polygon authentication failed: {response.text}"
            )

        if response.status_code == 429:
            raise RateLimitError("Polygon rate limit exceeded")

        if response.status_code != 200:
            raise ProviderError(
                f"Polygon API error: {response.status_code} - {response.text}"
            )

        return response.json()

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Polygon API.

        Uppercases the symbol.

        Args:
            symbol: Input symbol (e.g., "aapl" or "AAPL")

        Returns:
            Normalized symbol for API calls (e.g., "AAPL")
        """
        return symbol.upper()

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch OHLCV bar data from Polygon with automatic pagination.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
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

        timespan, multiplier = self.TIMEFRAME_MAP[timeframe]
        api_symbol = self._normalize_symbol(symbol)

        all_bars: list[dict[str, Any]] = []

        # Format dates as YYYY-MM-DD
        start_str = start.isoformat()
        end_str = end.isoformat()

        # Build endpoint
        endpoint = f"/v2/aggs/ticker/{api_symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"

        params: dict[str, Any] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,  # Polygon allows up to 50k per request
        }

        # Fetch data with pagination
        next_url: str | None = None
        while True:
            if next_url:
                # For pagination, use the full URL from next_url
                # But we need to make a direct request, not through _make_request
                client = self._get_client()
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
                try:
                    response = client.get(next_url, headers=headers)
                    if response.status_code in (401, 403):
                        raise AuthenticationError(
                            f"Polygon authentication failed: {response.text}"
                        )
                    if response.status_code == 429:
                        raise RateLimitError("Polygon rate limit exceeded")
                    if response.status_code != 200:
                        raise ProviderError(
                            f"Polygon API error: {response.status_code} - {response.text}"
                        )
                    data = response.json()
                except httpx.RequestError as e:
                    raise ProviderError(f"Polygon API request failed: {e}") from e
            else:
                data = self._make_request("GET", endpoint, params)

            results = data.get("results", [])
            if not results:
                break

            # Parse results
            for bar in results:
                ts = datetime.fromtimestamp(bar["t"] / 1000, tz=UTC)
                all_bars.append({
                    "timestamp": ts,
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar["v"]),
                })

            # Check for pagination
            next_url = data.get("next_url")
            if not next_url:
                break

        return self.bars_to_dataframe(all_bars)

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        """List available tickers from Polygon.

        Args:
            asset_class: Optional filter by asset class ("stocks", "crypto", etc.)

        Returns:
            DataFrame with instrument metadata
        """
        params: dict[str, Any] = {
            "active": "true",
            "limit": 1000,
        }

        # Map asset class to Polygon market type
        if asset_class:
            market_map = {
                "stocks": "stocks",
                "crypto": "crypto",
                "forex": "fx",
                "options": "options",
            }
            if asset_class in market_map:
                params["market"] = market_map[asset_class]

        data = self._make_request("GET", "/v3/reference/tickers", params)

        results = data.get("results", [])
        all_tickers: list[dict[str, Any]] = []

        for ticker in results:
            # Filter out inactive tickers
            if not ticker.get("active", True):
                continue

            all_tickers.append({
                "symbol": ticker.get("ticker", ""),
                "name": ticker.get("name", ""),
                "asset_class": ticker.get("market", "stocks"),
                "exchange": ticker.get("primary_exchange", ""),
                "type": ticker.get("type", ""),
                "currency": ticker.get("currency_name", "usd"),
            })

        if not all_tickers:
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "name": pl.Utf8,
                    "asset_class": pl.Utf8,
                    "exchange": pl.Utf8,
                    "type": pl.Utf8,
                    "currency": pl.Utf8,
                }
            )

        return pl.DataFrame(all_tickers)

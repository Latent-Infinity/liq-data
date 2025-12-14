"""Alpaca Markets API provider implementation.

This module implements the BaseProvider interface for Alpaca Markets API,
providing access to US equities market data.

Alpaca API Documentation: https://docs.alpaca.markets/docs/about-market-data-api

Features:
- US stocks data (NYSE, NASDAQ, etc.)
- OHLCV bar data via historical bars endpoint
- Header-based authentication (APCA-API-KEY-ID, APCA-API-SECRET-KEY)
- Rate limiting handling
- Automatic pagination for large datasets
"""

from datetime import date, datetime
from typing import Any

import httpx
import polars as pl

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from liq.data.providers.base import BaseProvider


class AlpacaProvider(BaseProvider):
    """Alpaca Markets API provider for market data.

    Provides access to US equities market data through Alpaca's REST API
    using header-based authentication.

    Example:
        provider = AlpacaProvider(
            api_key="your_api_key",
            api_secret="your_api_secret"
        )

        # Fetch stock data
        bars = provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            timeframe="1h"
        )
    """

    # Alpaca API URLs
    DATA_BASE_URL = "https://data.alpaca.markets"
    TRADING_BASE_URL = "https://api.alpaca.markets"

    # Timeframe mapping from our format to Alpaca format
    TIMEFRAME_MAP = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "1h": "1Hour",
        "4h": "4Hour",
        "1d": "1Day",
        "1w": "1Week",
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize Alpaca provider.

        Args:
            api_key: Alpaca API key (required)
            api_secret: Alpaca API secret (required)
            timeout: Request timeout in seconds (default: 30)

        Raises:
            ValueError: If api_key or api_secret is not provided
        """
        if not api_key:
            raise ValueError("api_key is required for Alpaca provider")
        if not api_secret:
            raise ValueError("api_secret is required for Alpaca provider")

        self._api_key = api_key
        self._api_secret = api_secret
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
        return "alpaca"

    @property
    def supported_timeframes(self) -> list[str]:
        """Return supported timeframes for Alpaca."""
        return list(self.TIMEFRAME_MAP.keys())

    def _make_request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
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
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Content-Type": "application/json",
        }

        try:
            response = client.request(method, url, params=params, headers=headers)
        except httpx.RequestError as e:
            raise ProviderError(f"Alpaca API request failed: {e}") from e

        if response.status_code in (401, 403):
            raise AuthenticationError(
                f"Alpaca authentication failed: {response.text}"
            )

        if response.status_code == 429:
            raise RateLimitError("Alpaca rate limit exceeded")

        if response.status_code != 200:
            raise ProviderError(
                f"Alpaca API error: {response.status_code} - {response.text}"
            )

        return response.json()

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Alpaca API.

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
        """Fetch OHLCV bar data from Alpaca with automatic pagination.

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

        alpaca_timeframe = self.TIMEFRAME_MAP[timeframe]
        api_symbol = self._normalize_symbol(symbol)

        all_bars: list[dict[str, Any]] = []

        # Format dates as RFC3339
        start_str = f"{start.isoformat()}T00:00:00Z"
        end_str = f"{end.isoformat()}T23:59:59Z"

        # Build URL
        url = f"{self.DATA_BASE_URL}/v2/stocks/{api_symbol}/bars"

        params: dict[str, Any] = {
            "start": start_str,
            "end": end_str,
            "timeframe": alpaca_timeframe,
            "limit": 10000,  # Alpaca max per request
            "adjustment": "all",  # Include all corporate action adjustments
        }

        # Fetch data with pagination
        page_token: str | None = None
        while True:
            if page_token:
                params["page_token"] = page_token
            elif "page_token" in params:
                del params["page_token"]

            data = self._make_request("GET", url, params)

            bars = data.get("bars")
            if not bars:
                break

            # Parse bars
            for bar in bars:
                # Alpaca timestamps are ISO 8601 format
                ts = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                all_bars.append({
                    "timestamp": ts,
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar["v"]),
                })

            # Check for pagination
            page_token = data.get("next_page_token")
            if not page_token:
                break

        return self.bars_to_dataframe(all_bars)

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:  # noqa: ARG002
        """List available assets from Alpaca.

        Args:
            asset_class: Optional filter by asset class (not used, always us_equity)

        Returns:
            DataFrame with instrument metadata
        """
        url = f"{self.TRADING_BASE_URL}/v2/assets"

        params: dict[str, Any] = {
            "status": "active",
        }

        data = self._make_request("GET", url, params)

        all_assets: list[dict[str, Any]] = []

        for asset in data:
            # Filter out inactive assets
            if asset.get("status") != "active":
                continue

            all_assets.append({
                "symbol": asset.get("symbol", ""),
                "name": asset.get("name", ""),
                "asset_class": asset.get("class", "us_equity"),
                "exchange": asset.get("exchange", ""),
                "tradable": asset.get("tradable", False),
                "fractionable": asset.get("fractionable", False),
            })

        if not all_assets:
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "name": pl.Utf8,
                    "asset_class": pl.Utf8,
                    "exchange": pl.Utf8,
                    "tradable": pl.Boolean,
                    "fractionable": pl.Boolean,
                }
            )

        return pl.DataFrame(all_assets)

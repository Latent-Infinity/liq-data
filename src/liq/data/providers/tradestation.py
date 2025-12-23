"""TradeStation REST API provider implementation.

This module implements the BaseProvider interface for TradeStation's brokerage API,
providing access to stocks (equities) and futures market data.

TradeStation API Documentation: https://api.tradestation.com/docs/

Features:
- Stock and futures instrument discovery
- OHLCV bar data via GetBars endpoint
- OAuth2 authentication with automatic token refresh
- Rate limiting (120 requests/minute)
- Automatic pagination for large date ranges
"""

from datetime import UTC, date, datetime, timedelta
import logging
from typing import Any, cast
from urllib.parse import urlencode

import httpx
import polars as pl

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from liq.data.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class TradeStationProvider(BaseProvider):
    """TradeStation REST API provider for stocks and futures data.

    Provides access to TradeStation's market data through their REST API
    using OAuth2 authentication.

    Example:
        provider = TradeStationProvider(
            client_id="your_client_id",
            client_secret="your_client_secret",
            refresh_token="your_refresh_token",
        )

        # Fetch stock data
        bars = provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            timeframe="1h"
        )

        # Fetch futures data
        bars = provider.fetch_bars(
            "ESZ24",
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            timeframe="1h"
        )
    """

    # TradeStation API URLs
    BASE_URL = "https://api.tradestation.com/v3"
    AUTH_URL = "https://signin.tradestation.com/oauth/token"
    AUTHORIZE_URL = "https://signin.tradestation.com/authorize"
    DEFAULT_AUDIENCE = "https://api.tradestation.com"
    DEFAULT_SCOPE = "openid profile offline_access MarketData ReadAccount Trade"

    # Timeframe to TradeStation unit/interval mapping
    TIMEFRAME_MAP = {
        "1m": ("Minute", 1),
        "5m": ("Minute", 5),
        "15m": ("Minute", 15),
        "30m": ("Minute", 30),
        "1h": ("Minute", 60),
        "4h": ("Minute", 240),
        "1d": ("Daily", 1),
        "1w": ("Weekly", 1),
    }

    # Maximum bars per request (TradeStation limit)
    MAX_BARS_PER_REQUEST = 57600  # ~40 days of minute data

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize TradeStation provider.

        Args:
            client_id: OAuth2 client ID (required)
            client_secret: OAuth2 client secret (required)
            refresh_token: OAuth2 refresh token (required)
            timeout: Request timeout in seconds (default: 30)

        Raises:
            ValueError: If required credentials are missing
        """
        if not client_id:
            raise ValueError("client_id is required for TradeStation provider")
        if not client_secret:
            raise ValueError("client_secret is required for TradeStation provider")
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._timeout = timeout

        # Token state
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

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
        return "tradestation"

    @property
    def supported_timeframes(self) -> list[str]:
        """Return supported timeframes for TradeStation."""
        return list(self.TIMEFRAME_MAP.keys())

    @property
    def refresh_token(self) -> str | None:
        """Return the current refresh token (may rotate)."""
        return self._refresh_token

    @classmethod
    def build_authorization_url(
        cls,
        client_id: str,
        redirect_uri: str,
        scope: str | None = None,
        state: str | None = None,
        audience: str | None = None,
    ) -> str:
        """Build the TradeStation authorization URL for the Auth Code flow."""
        params = {
            "response_type": "code",
            "client_id": client_id,
            "audience": audience or cls.DEFAULT_AUDIENCE,
            "redirect_uri": redirect_uri,
            "scope": scope or cls.DEFAULT_SCOPE,
        }
        if state:
            params["state"] = state
        return f"{cls.AUTHORIZE_URL}?{urlencode(params)}"

    @classmethod
    def exchange_authorization_code(
        cls,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Exchange an authorization code for access and refresh tokens."""
        try:
            response = httpx.post(
                cls.AUTH_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=timeout,
            )
        except httpx.RequestError as e:
            raise AuthenticationError(f"OAuth2 auth-code exchange failed: {e}") from e

        if response.status_code != 200:
            raise AuthenticationError(
                f"OAuth2 auth-code exchange failed: {response.status_code} - {response.text}"
            )

        return cast(dict[str, Any], response.json())

    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid access token, refreshing if needed."""
        now = datetime.now(UTC)

        # Check if we need to refresh - skip if token valid for 60+ seconds
        if (
            self._access_token
            and self._token_expires_at
            and now < self._token_expires_at - timedelta(seconds=60)
        ):
            return

        if not self._refresh_token:
            raise AuthenticationError(
                "TradeStation refresh token not configured. "
                "Generate one via the auth code flow and set TRADESTATION_REFRESH_TOKEN."
            )

        # Refresh the token
        client = self._get_client()

        try:
            response = client.post(
                self.AUTH_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": self._refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.RequestError as e:
            raise AuthenticationError(f"OAuth2 token refresh request failed: {e}") from e

        if response.status_code != 200:
            raise AuthenticationError(
                f"OAuth2 token refresh failed: {response.status_code} - {response.text}"
            )

        data = response.json()
        self._access_token = data["access_token"]
        self._token_expires_at = now + timedelta(seconds=data.get("expires_in", 1200))

        # Update refresh token if a new one was issued
        if "refresh_token" in data:
            new_refresh = data["refresh_token"]
            if new_refresh and new_refresh != self._refresh_token:
                from liq.data.settings import get_settings, persist_env_value

                settings = get_settings()
                if settings.tradestation_persist_refresh_token:
                    persist_env_value("TRADESTATION_REFRESH_TOKEN", new_refresh)
                    logger.info(
                        "TradeStation refresh token rotated and persisted to .env."
                    )
                else:
                    logger.warning(
                        "TradeStation refresh token rotated; update TRADESTATION_REFRESH_TOKEN "
                        "to persist it across runs."
                    )
                self._refresh_token = new_refresh

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response as dict

        Raises:
            AuthenticationError: If authentication fails
            RateLimitError: If rate limit is exceeded
            ProviderError: If API call fails
        """
        self._ensure_authenticated()

        client = self._get_client()
        url = f"{self.BASE_URL}{endpoint}"

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        try:
            response = client.request(method, url, params=params, headers=headers)
        except httpx.RequestError as e:
            raise ProviderError(f"TradeStation API request failed: {e}") from e

        if response.status_code == 401:
            raise AuthenticationError(f"TradeStation authentication failed: {response.text}")

        if response.status_code == 429:
            raise RateLimitError("TradeStation rate limit exceeded")

        if response.status_code != 200:
            raise ProviderError(
                f"TradeStation API error: {response.status_code} - {response.text}"
            )

        return cast(dict[str, Any], response.json())

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for TradeStation API.

        For continuous futures, prefix with @.
        Stock symbols are used as-is.

        Args:
            symbol: Input symbol

        Returns:
            Normalized symbol for API calls
        """
        # Common continuous futures symbols
        continuous_futures = ["ES", "NQ", "CL", "GC", "SI", "ZB", "ZN", "ZS", "ZC", "ZW"]

        if symbol.upper() in continuous_futures:
            return f"@{symbol.upper()}"

        return symbol.upper()

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch OHLCV bar data from TradeStation with automatic pagination.

        Args:
            symbol: Stock symbol (e.g., "AAPL") or futures (e.g., "ESZ24")
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Bar timeframe (default: "1d")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ProviderError: If API call fails
            RateLimitError: If rate limit is exceeded
        """
        self.validate_timeframe(timeframe)

        unit, interval = self.TIMEFRAME_MAP[timeframe]
        api_symbol = self._normalize_symbol(symbol)

        all_bars: list[dict[str, Any]] = []
        current_end = datetime.combine(end, datetime.max.time()).replace(tzinfo=UTC)
        start_dt = datetime.combine(start, datetime.min.time()).replace(tzinfo=UTC)

        while current_end > start_dt:
            # Build params
            params: dict[str, Any] = {
                "unit": unit,
                "interval": str(interval),
                "barsback": str(self.MAX_BARS_PER_REQUEST),
                "lastdate": current_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            data = self._make_request("GET", f"/marketdata/barcharts/{api_symbol}", params)

            bars = data.get("Bars", [])
            if not bars:
                break

            # Parse and filter bars
            for bar in bars:
                ts = datetime.fromisoformat(bar["TimeStamp"].replace("Z", "+00:00"))
                if ts < start_dt:
                    continue

                all_bars.append({
                    "timestamp": ts,
                    "open": float(bar["Open"]),
                    "high": float(bar["High"]),
                    "low": float(bar["Low"]),
                    "close": float(bar["Close"]),
                    "volume": float(bar.get("TotalVolume", 0)),
                })

            # Find earliest timestamp for next pagination
            earliest = min(
                datetime.fromisoformat(b["TimeStamp"].replace("Z", "+00:00"))
                for b in bars
            )

            # If we've gotten all the data we need, stop
            if earliest <= start_dt:
                break

            # Move to earlier period
            current_end = earliest - timedelta(seconds=1)

        return self.bars_to_dataframe(all_bars)

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        """List available instruments from TradeStation.

        Args:
            asset_class: Optional filter: "stocks" or "futures"

        Returns:
            DataFrame with instrument metadata
        """
        # TradeStation doesn't have a comprehensive symbol search endpoint
        # We'll use a search with common patterns
        search_terms = []

        if asset_class == "stocks" or asset_class is None:
            search_terms.append("STOCK")

        if asset_class == "futures" or asset_class is None:
            search_terms.append("FUTURE")

        all_symbols: list[dict[str, Any]] = []

        for search_type in search_terms:
            try:
                # Use symbol suggest endpoint
                params = {"$top": "100", "category": search_type}
                data = self._make_request("GET", "/marketdata/symbols", params)

                symbols = data.get("Symbols", [])
                for sym in symbols:
                    asset_type = sym.get("AssetType", "UNKNOWN")

                    # Filter by asset class if specified
                    if asset_class == "stocks" and asset_type != "STOCK":
                        continue
                    if asset_class == "futures" and asset_type != "FUTURE":
                        continue

                    all_symbols.append({
                        "symbol": sym.get("Symbol", ""),
                        "name": sym.get("Description", ""),
                        "asset_class": asset_type,
                        "exchange": sym.get("Exchange", ""),
                        "type": asset_type,
                    })
            except ProviderError:
                # Continue if search fails
                continue

        if not all_symbols:
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "name": pl.Utf8,
                    "asset_class": pl.Utf8,
                    "exchange": pl.Utf8,
                    "type": pl.Utf8,
                }
            )

        return pl.DataFrame(all_symbols)

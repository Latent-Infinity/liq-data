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

import asyncio
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import httpx
import polars as pl

from liq.data.alpaca_action_schema import AlpacaActionSchemaRequest, validate_action_schema_request
from liq.data.alpaca_announcement_source import (
    AlpacaAnnouncementCalendarRequest,
    AlpacaAnnouncementDailyRequest,
    validate_announcement_request,
)
from liq.data.alpaca_event_basis_source import (
    AlpacaEventBasisDailyRequest,
    validate_event_basis_request,
)
from liq.data.alpaca_momentum_source import (
    AlpacaMomentumCalendarRequest,
    AlpacaMomentumDailyRequest,
)
from liq.data.alpaca_source import (
    SOURCE_PAGE_TIMEOUT_SECONDS,
    AlpacaBarsRequest,
    AlpacaCalendarRequest,
    AlpacaCaptureRefused,
    AlpacaCorporateActionsRequest,
    AlpacaDailyFixtureRequest,
)
from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from liq.data.providers.base import BaseProvider


class _AttemptTransport(httpx.AsyncBaseTransport):
    """Witness an initiated transport call, including connection refusals."""

    def __init__(self, delegate: httpx.AsyncBaseTransport, witness: Callable[[], None]) -> None:
        self.delegate = delegate
        self.witness = witness

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.witness()
        return await self.delegate.handle_async_request(request)

    async def aclose(self) -> None:
        await self.delegate.aclose()


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
        self._source_transport: httpx.AsyncBaseTransport | None = None

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
            raise AuthenticationError(f"Alpaca authentication failed: {response.text}")

        if response.status_code == 429:
            raise RateLimitError("Alpaca rate limit exceeded")

        if response.status_code != 200:
            raise ProviderError(f"Alpaca API error: {response.status_code} - {response.text}")

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
                all_bars.append(
                    {
                        "timestamp": ts,
                        "open": float(bar["o"]),
                        "high": float(bar["h"]),
                        "low": float(bar["l"]),
                        "close": float(bar["c"]),
                        "volume": float(bar["v"]),
                    }
                )

            # Check for pagination
            page_token = data.get("next_page_token")
            if not page_token:
                break

        return self.bars_to_dataframe(all_bars)

    def get_corporate_actions(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """Fetch complete-quality corporate actions for one US equity.

        Alpaca groups heterogeneous action records by type. The provider
        flattens those groups while retaining the group name in
        ``corporate_action_type`` so callers can distinguish splits,
        distributions, and other actions without relying on row shape.

        The endpoint filters and sorts on ``process_date``; consumers that
        need ex-date coverage must evaluate the returned date fields and must
        not interpret an empty response as proof of source completeness.
        """
        api_symbol = self._normalize_symbol(symbol)
        url = f"{self.DATA_BASE_URL}/v1/corporate-actions"
        params: dict[str, Any] = {
            "symbols": api_symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
            "data_quality": "complete",
            "sort": "asc",
        }

        actions: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            if page_token:
                params["page_token"] = page_token
            else:
                params.pop("page_token", None)

            data = self._make_request("GET", url, params)
            grouped = data.get("corporate_actions") if isinstance(data, dict) else None
            if not isinstance(grouped, dict):
                raise ProviderError("Alpaca returned a malformed corporate-actions response")

            for action_type, records in grouped.items():
                if not isinstance(records, list):
                    raise ProviderError("Alpaca returned a malformed corporate-actions response")
                for record in records:
                    if not isinstance(record, dict):
                        raise ProviderError(
                            "Alpaca returned a malformed corporate-actions response"
                        )
                    normalized = dict(record)
                    normalized.setdefault("corporate_action_type", action_type)
                    actions.append(normalized)

            page_token = data.get("next_page_token")
            if not page_token:
                break

        return actions

    def fetch_source_page(
        self,
        request: AlpacaBarsRequest
        | AlpacaCorporateActionsRequest
        | AlpacaActionSchemaRequest
        | AlpacaDailyFixtureRequest
        | AlpacaCalendarRequest
        | AlpacaMomentumCalendarRequest
        | AlpacaMomentumDailyRequest
        | AlpacaAnnouncementCalendarRequest
        | AlpacaAnnouncementDailyRequest
        | AlpacaEventBasisDailyRequest,
        *,
        page_token: str | None,
        on_http_attempt: Callable[[], None] | None = None,
    ) -> bytes:
        if isinstance(request, AlpacaActionSchemaRequest):
            request = validate_action_schema_request(request)
            if page_token is not None:
                raise AlpacaCaptureRefused("action_schema_continuation_forbidden")
        if isinstance(request, AlpacaEventBasisDailyRequest):
            request = validate_event_basis_request(request)
        if isinstance(request, (AlpacaAnnouncementCalendarRequest, AlpacaAnnouncementDailyRequest)):
            request = validate_announcement_request(request)
        if isinstance(request, AlpacaMomentumCalendarRequest):
            request = AlpacaMomentumCalendarRequest.model_validate(request)
        elif isinstance(request, AlpacaMomentumDailyRequest):
            request = AlpacaMomentumDailyRequest.model_validate(request)
        if isinstance(
            request,
            (AlpacaCalendarRequest, AlpacaMomentumCalendarRequest, AlpacaAnnouncementCalendarRequest),
        ):
            url = "https://paper-api.alpaca.markets/v2/calendar"
            params = {
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "date_type": "TRADING",
            }
            max_bytes = request.max_response_bytes
        elif isinstance(request, (AlpacaCorporateActionsRequest, AlpacaActionSchemaRequest)):
            url = f"{self.DATA_BASE_URL}/v1/corporate-actions"
            params: dict[str, Any] = {
                "symbols": ",".join(request.symbols),
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "data_quality": request.data_quality,
                "limit": request.limit,
                "sort": "asc",
                "region": "us",
            }
            max_bytes = request.max_response_bytes
        elif isinstance(
            request,
            (AlpacaBarsRequest, AlpacaDailyFixtureRequest, AlpacaMomentumDailyRequest, AlpacaAnnouncementDailyRequest, AlpacaEventBasisDailyRequest),
        ):
            url = f"{self.DATA_BASE_URL}/v2/stocks/bars"
            params = {
                "symbols": request.symbol,
                "start": request.start.isoformat().replace("+00:00", "Z"),
                "end": request.end.isoformat().replace("+00:00", "Z"),
                "timeframe": request.timeframe,
                "adjustment": request.adjustment,
                "asof": request.asof.isoformat(),
                "feed": request.feed,
                "currency": request.currency,
                "limit": request.limit,
                "sort": "asc",
            }
            max_bytes = request.max_response_bytes
        else:
            raise TypeError("unsupported Alpaca source request")
        if page_token is not None:
            if isinstance(request, (AlpacaCalendarRequest, AlpacaMomentumCalendarRequest, AlpacaAnnouncementCalendarRequest)):
                raise AlpacaCaptureRefused("calendar_continuation_forbidden")
            if isinstance(request, (AlpacaDailyFixtureRequest, AlpacaMomentumDailyRequest, AlpacaAnnouncementDailyRequest, AlpacaEventBasisDailyRequest)):
                raise AlpacaCaptureRefused("daily_fixture_continuation_forbidden")
            if not isinstance(page_token, str) or not page_token or len(page_token) > 4096:
                raise AlpacaCaptureRefused("invalid_page_token")
            params["page_token"] = page_token

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        }
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise AlpacaCaptureRefused("source_page_requires_sync_context")

        async def read_page() -> bytes:
            transport = self._source_transport
            if on_http_attempt is not None:
                transport = _AttemptTransport(
                    transport or httpx.AsyncHTTPTransport(retries=0), on_http_attempt
                )
            async with (
                httpx.AsyncClient(
                    timeout=httpx.Timeout(SOURCE_PAGE_TIMEOUT_SECONDS),
                    follow_redirects=False,
                    transport=transport,
                ) as client,
                client.stream(
                    "GET",
                    url,
                    params=params,
                    headers=headers,
                    follow_redirects=False,
                ) as response,
            ):
                if response.status_code != 200:
                    raise AlpacaCaptureRefused("http_refused", response.status_code)
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise AlpacaCaptureRefused("unexpected_content_encoding", response.status_code)
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                if content_type != "application/json":
                    raise AlpacaCaptureRefused("unexpected_content_type", response.status_code)
                body = bytearray()
                async for chunk in response.aiter_raw(chunk_size=65536):
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise AlpacaCaptureRefused("response_byte_limit", response.status_code)
                return bytes(body)

        try:
            return asyncio.run(asyncio.wait_for(read_page(), timeout=SOURCE_PAGE_TIMEOUT_SECONDS))
        except TimeoutError as exc:
            raise AlpacaCaptureRefused("request_deadline") from exc
        except AlpacaCaptureRefused:
            raise
        except httpx.HTTPError as exc:
            raise AlpacaCaptureRefused("transport_error") from exc

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

            all_assets.append(
                {
                    "symbol": asset.get("symbol", ""),
                    "name": asset.get("name", ""),
                    "asset_class": asset.get("class", "us_equity"),
                    "exchange": asset.get("exchange", ""),
                    "tradable": asset.get("tradable", False),
                    "fractionable": asset.get("fractionable", False),
                }
            )

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

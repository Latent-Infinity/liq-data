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
from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_instrument_metadata import InstrumentMetadataObservation
from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.oanda_source_metadata import CandleDay, CandleDayObservation
from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE, BaseProvider
from liq.data.providers.oanda_account_probe import AccountCandleObservation, probe_account_candle
from liq.data.providers.oanda_candles import PriceComponents, fetch_candle_chunk
from liq.data.providers.oanda_price_capture import PriceCaptureFailure, capture_source_prices
from liq.data.providers.oanda_source_body import SourceBody
from liq.data.providers.oanda_source_transport import probe_instrument_metadata, probe_source_day


class OandaProviderClosed(ProviderError):
    """A terminal provider cannot allocate or expose a client."""

    def __init__(self) -> None:
        super().__init__("PROVIDER_CLOSED")


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
        self._closed = False

    def _get_client(self) -> httpx.Client:
        """Get or create the HTTP client."""
        if self._closed:
            raise OandaProviderClosed()
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
        return self.bars_to_dataframe(self._fetch_candles(symbol, start, end, timeframe, "M"))

    def fetch_bid_ask_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1m",
    ) -> pl.DataFrame:
        """Fetch historical base-price bid/ask OHLC candles, not account-specific fills."""
        columns = [
            "timestamp",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
            "volume",
        ]
        rows = self._fetch_candles(symbol, start, end, timeframe, "BA")
        schema = {"timestamp": pl.Datetime("us", "UTC")}
        schema.update(dict.fromkeys(columns[1:-1], PRICE_DTYPE))
        schema["volume"] = VOLUME_DTYPE
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    def probe_account_candle(self) -> AccountCandleObservation:
        """Return metadata from the separate, fixed account-candle compatibility route."""
        return probe_account_candle(self._get_client(), self._base_url, self._account_id)

    def probe_instrument_metadata(self) -> InstrumentMetadataObservation:
        """Expose only selected current membership and native quote-unit metadata."""
        return probe_instrument_metadata(self)

    def probe_source_day(self, scope: CandleDay) -> CandleDayObservation:
        """Expose only coverage metadata for a fixed account-scoped daily case."""
        return probe_source_day(self, scope)

    def capture_source_prices(self, scope: SourcePriceWindow) -> SourceBody | PriceCaptureFailure:
        """Transport-only capability; research collectors must use the guarded service."""
        return capture_source_prices(self, scope)

    def capture_cohort_source_prices(
        self, scope: CohortSourceWindow
    ) -> SourceBody | PriceCaptureFailure:
        """Expose the distinct fixed-cohort capability without granting research access."""
        from liq.data.providers.oanda_cohort_prices import capture_cohort_source_prices

        return capture_cohort_source_prices(self, scope)

    def _fetch_candles(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str,
        price_components: PriceComponents,
    ) -> list[dict[str, Any]]:
        """Paginate one historical price basis without repairing missing candles."""
        self.validate_timeframe(timeframe)
        granularity = self.TIMEFRAME_MAP[timeframe]
        minutes_per_candle = self.TIMEFRAME_MINUTES[timeframe]
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=UTC)
        chunk_delta = timedelta(minutes=self.MAX_CANDLES_PER_REQUEST * minutes_per_candle)
        all_bars: list[dict[str, Any]] = []
        current_start = start_dt
        try:
            while current_start < end_dt:
                current_end = min(current_start + chunk_delta, end_dt)
                chunk_bars = self._fetch_bars_chunk(
                    symbol, current_start, current_end, granularity, price_components
                )
                all_bars.extend(chunk_bars)
                if chunk_bars:
                    last_timestamp = chunk_bars[-1]["timestamp"]
                    current_start = last_timestamp + timedelta(minutes=minutes_per_candle)
                else:
                    current_start = current_end
            return all_bars
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
        price_components: PriceComponents = "M",
    ) -> list[dict[str, Any]]:
        """Fetch one chunk, preserving the requested price basis."""
        return fetch_candle_chunk(
            self._get_client(),
            self._base_url,
            symbol,
            start_dt,
            end_dt,
            granularity,
            price_components,
        )

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
                instruments.append(
                    {
                        "symbol": inst["name"],
                        "name": inst.get("displayName", inst["name"]),
                        "asset_class": "forex",
                        "type": inst.get("type", "CURRENCY"),
                    }
                )

            return pl.DataFrame(instruments)

        except (AuthenticationError, RateLimitError):
            raise
        except (httpx.RequestError, KeyError, ValueError) as e:
            raise ProviderError(f"Failed to fetch OANDA instruments: {e}") from e

    def fetch_instruments(self, asset_class: str) -> list[Any]:
        return self.list_instruments(asset_class).to_dicts()

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

    def close(self) -> None:
        """Make shutdown terminal before closing the owned client, without retries."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            self._client.close()

    def __del__(self) -> None:
        """Close an initialized provider only if shutdown has not been attempted."""
        if not getattr(self, "_closed", True):
            self.close()

"""Tests for liq.data.protocols module."""

from datetime import date

import polars as pl

from liq.data.protocols import AsyncMarketDataProvider, MarketDataProvider
from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE


class TestMarketDataProviderProtocol:
    """Tests for MarketDataProvider protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should be runtime checkable."""
        assert hasattr(MarketDataProvider, "__protocol_attrs__") or isinstance(MarketDataProvider, type)

    def test_protocol_defines_name(self) -> None:
        """Protocol should define name property."""
        assert hasattr(MarketDataProvider, "name")

    def test_protocol_defines_fetch_bars(self) -> None:
        """Protocol should define fetch_bars method."""
        assert hasattr(MarketDataProvider, "fetch_bars")

    def test_protocol_defines_list_instruments(self) -> None:
        """Protocol should define list_instruments method."""
        assert hasattr(MarketDataProvider, "list_instruments")

    def test_protocol_defines_extended_surface(self) -> None:
        """Protocol should define extended PRD methods."""
        assert hasattr(MarketDataProvider, "get_instrument")
        assert hasattr(MarketDataProvider, "fetch_quotes")
        assert hasattr(MarketDataProvider, "fetch_fundamentals")
        assert hasattr(MarketDataProvider, "get_corporate_actions")
        assert hasattr(MarketDataProvider, "get_universe")
        assert hasattr(MarketDataProvider, "fetch_instruments")
        assert hasattr(MarketDataProvider, "validate_credentials")
        assert hasattr(MarketDataProvider, "supported_asset_classes")


class MockMarketDataProvider:
    """Mock implementation for testing protocol conformance."""

    def __init__(self) -> None:
        self._instruments = pl.DataFrame(
            {
                "symbol": ["EUR_USD", "GBP_USD"],
                "name": ["Euro/USD", "British Pound/USD"],
            }
        )

    @property
    def name(self) -> str:
        return "mock_provider"

    def fetch_bars(  # noqa: ARG002
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "open": PRICE_DTYPE,
                "high": PRICE_DTYPE,
                "low": PRICE_DTYPE,
                "close": PRICE_DTYPE,
                "volume": VOLUME_DTYPE,
            }
        )

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:  # noqa: ARG002
        return self._instruments

    def get_instrument(self, symbol: str) -> dict | None:
        filtered = self._instruments.filter(pl.col("symbol") == symbol)
        if filtered.is_empty():
            return None
        return filtered.row(0, named=True)

    @property
    def supported_asset_classes(self) -> list[str]:
        return ["forex"]

    def validate_credentials(self) -> bool:
        return True

    def fetch_quotes(self, symbol: str, start: date, end: date) -> pl.DataFrame:  # noqa: ARG002
        return pl.DataFrame({"timestamp": [], "bid": [], "ask": []})

    def fetch_fundamentals(self, symbol: str, as_of: date) -> dict:  # noqa: ARG002
        return {"symbol": symbol}

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[dict]:  # noqa: ARG002
        return []

    def get_universe(self, asset_class: str, as_of: date | None = None) -> list[str]:  # noqa: ARG002
        return ["EUR_USD", "GBP_USD"]

    def fetch_instruments(self, asset_class: str) -> list[dict]:  # noqa: ARG002
        return [{"symbol": "EUR_USD"}]


class TestMockMarketDataProvider:
    """Tests for mock implementation to verify protocol works."""

    def test_mock_implements_protocol(self) -> None:
        """Mock should satisfy the protocol structure."""
        provider = MockMarketDataProvider()
        assert isinstance(provider, MarketDataProvider)

    def test_mock_name(self) -> None:
        provider = MockMarketDataProvider()
        assert provider.name == "mock_provider"

    def test_mock_fetch_bars_returns_dataframe(self) -> None:
        provider = MockMarketDataProvider()
        result = provider.fetch_bars("EUR_USD", date(2024, 1, 1), date(2024, 1, 31))
        assert isinstance(result, pl.DataFrame)

    def test_mock_list_instruments_returns_dataframe(self) -> None:
        provider = MockMarketDataProvider()
        result = provider.list_instruments()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_mock_get_instrument_returns_dict(self) -> None:
        provider = MockMarketDataProvider()
        result = provider.get_instrument("EUR_USD")
        assert result is not None
        assert result["symbol"] == "EUR_USD"

    def test_mock_get_instrument_returns_none_for_unknown(self) -> None:
        provider = MockMarketDataProvider()
        result = provider.get_instrument("UNKNOWN")
        assert result is None

    def test_mock_supports_extended_methods(self) -> None:
        provider = MockMarketDataProvider()
        assert provider.validate_credentials()
        assert provider.supported_asset_classes == ["forex"]
        assert provider.fetch_quotes("EUR_USD", date(2024, 1, 1), date(2024, 1, 2)).is_empty()


class TestAsyncMarketDataProviderProtocol:
    """Tests for AsyncMarketDataProvider protocol."""

    def test_async_protocol_is_runtime_checkable(self) -> None:
        """Async protocol should be runtime checkable."""
        assert hasattr(AsyncMarketDataProvider, "__protocol_attrs__") or isinstance(AsyncMarketDataProvider, type)

    def test_async_protocol_defines_name(self) -> None:
        """Async protocol should define name property."""
        assert hasattr(AsyncMarketDataProvider, "name")

    def test_async_protocol_defines_fetch_bars(self) -> None:
        """Async protocol should define fetch_bars method."""
        assert hasattr(AsyncMarketDataProvider, "fetch_bars")

    def test_async_protocol_defines_list_instruments(self) -> None:
        """Async protocol should define list_instruments method."""
        assert hasattr(AsyncMarketDataProvider, "list_instruments")

    def test_async_protocol_defines_extended_surface(self) -> None:
        assert hasattr(AsyncMarketDataProvider, "fetch_quotes")
        assert hasattr(AsyncMarketDataProvider, "fetch_fundamentals")
        assert hasattr(AsyncMarketDataProvider, "get_corporate_actions")
        assert hasattr(AsyncMarketDataProvider, "get_universe")
        assert hasattr(AsyncMarketDataProvider, "fetch_instruments")
        assert hasattr(AsyncMarketDataProvider, "validate_credentials")
        assert hasattr(AsyncMarketDataProvider, "supported_asset_classes")

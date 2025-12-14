"""Tests for liq.data.protocols module."""

from datetime import date

import polars as pl

from liq.data.protocols import AsyncDataProvider, DataProvider
from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE


class TestDataProviderProtocol:
    """Tests for DataProvider protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should be runtime checkable."""
        assert hasattr(DataProvider, "__protocol_attrs__") or isinstance(DataProvider, type)

    def test_protocol_defines_name(self) -> None:
        """Protocol should define name property."""
        assert hasattr(DataProvider, "name")

    def test_protocol_defines_fetch_bars(self) -> None:
        """Protocol should define fetch_bars method."""
        assert hasattr(DataProvider, "fetch_bars")

    def test_protocol_defines_list_instruments(self) -> None:
        """Protocol should define list_instruments method."""
        assert hasattr(DataProvider, "list_instruments")

    def test_protocol_defines_extended_surface(self) -> None:
        """Protocol should define extended PRD methods."""
        assert hasattr(DataProvider, "get_instrument")
        assert hasattr(DataProvider, "fetch_quotes")
        assert hasattr(DataProvider, "fetch_fundamentals")
        assert hasattr(DataProvider, "get_corporate_actions")
        assert hasattr(DataProvider, "get_universe")
        assert hasattr(DataProvider, "fetch_instruments")
        assert hasattr(DataProvider, "validate_credentials")
        assert hasattr(DataProvider, "supported_asset_classes")


class MockDataProvider:
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


class TestMockDataProvider:
    """Tests for mock implementation to verify protocol works."""

    def test_mock_implements_protocol(self) -> None:
        """Mock should satisfy the protocol structure."""
        provider = MockDataProvider()
        assert isinstance(provider, DataProvider)

    def test_mock_name(self) -> None:
        provider = MockDataProvider()
        assert provider.name == "mock_provider"

    def test_mock_fetch_bars_returns_dataframe(self) -> None:
        provider = MockDataProvider()
        result = provider.fetch_bars("EUR_USD", date(2024, 1, 1), date(2024, 1, 31))
        assert isinstance(result, pl.DataFrame)

    def test_mock_list_instruments_returns_dataframe(self) -> None:
        provider = MockDataProvider()
        result = provider.list_instruments()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_mock_get_instrument_returns_dict(self) -> None:
        provider = MockDataProvider()
        result = provider.get_instrument("EUR_USD")
        assert result is not None
        assert result["symbol"] == "EUR_USD"

    def test_mock_get_instrument_returns_none_for_unknown(self) -> None:
        provider = MockDataProvider()
        result = provider.get_instrument("UNKNOWN")
        assert result is None

    def test_mock_supports_extended_methods(self) -> None:
        provider = MockDataProvider()
        assert provider.validate_credentials()
        assert provider.supported_asset_classes == ["forex"]
        assert provider.fetch_quotes("EUR_USD", date(2024, 1, 1), date(2024, 1, 2)).is_empty()


class TestAsyncDataProviderProtocol:
    """Tests for AsyncDataProvider protocol."""

    def test_async_protocol_is_runtime_checkable(self) -> None:
        """Async protocol should be runtime checkable."""
        assert hasattr(AsyncDataProvider, "__protocol_attrs__") or isinstance(
            AsyncDataProvider, type
        )

    def test_async_protocol_defines_name(self) -> None:
        """Async protocol should define name property."""
        assert hasattr(AsyncDataProvider, "name")

    def test_async_protocol_defines_fetch_bars(self) -> None:
        """Async protocol should define fetch_bars method."""
        assert hasattr(AsyncDataProvider, "fetch_bars")

    def test_async_protocol_defines_list_instruments(self) -> None:
        """Async protocol should define list_instruments method."""
        assert hasattr(AsyncDataProvider, "list_instruments")

    def test_async_protocol_defines_extended_surface(self) -> None:
        assert hasattr(AsyncDataProvider, "fetch_quotes")
        assert hasattr(AsyncDataProvider, "fetch_fundamentals")
        assert hasattr(AsyncDataProvider, "get_corporate_actions")
        assert hasattr(AsyncDataProvider, "get_universe")
        assert hasattr(AsyncDataProvider, "fetch_instruments")
        assert hasattr(AsyncDataProvider, "validate_credentials")
        assert hasattr(AsyncDataProvider, "supported_asset_classes")

"""Tests for liq.data.providers module."""

from datetime import date

import polars as pl
import pytest

from liq.data.exceptions import ProviderError
from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE, BaseProvider


class ConcreteProvider(BaseProvider):
    """Concrete implementation for testing BaseProvider."""

    def __init__(self, provider_name: str = "test_provider") -> None:
        self._name = provider_name
        self._instruments = pl.DataFrame(
            {
                "symbol": ["EUR_USD", "GBP_USD", "USD_JPY"],
                "name": ["Euro/USD", "British Pound/USD", "USD/Yen"],
                "asset_class": ["forex", "forex", "forex"],
            }
        )

    @property
    def name(self) -> str:
        return self._name

    def fetch_bars(  # noqa: ARG002
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        self.validate_timeframe(timeframe)
        # Return empty DataFrame for testing
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

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        if asset_class is None:
            return self._instruments
        return self._instruments.filter(pl.col("asset_class") == asset_class)


class TestBaseProviderCreation:
    """Tests for BaseProvider instantiation."""

    def test_create_concrete_provider(self) -> None:
        provider = ConcreteProvider()
        assert provider.name == "test_provider"

    def test_create_provider_with_custom_name(self) -> None:
        provider = ConcreteProvider("custom_provider")
        assert provider.name == "custom_provider"


class TestBaseProviderTimeframes:
    """Tests for BaseProvider timeframe handling."""

    def test_standard_timeframes(self) -> None:
        provider = ConcreteProvider()
        assert "1m" in provider.supported_timeframes
        assert "1h" in provider.supported_timeframes
        assert "1d" in provider.supported_timeframes

    def test_validate_timeframe_valid(self) -> None:
        provider = ConcreteProvider()
        # Should not raise
        provider.validate_timeframe("1d")

    def test_validate_timeframe_invalid_raises(self) -> None:
        provider = ConcreteProvider()
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            provider.validate_timeframe("invalid")


class TestBaseProviderFetchBars:
    """Tests for BaseProvider.fetch_bars method."""

    def test_fetch_bars_returns_dataframe(self) -> None:
        provider = ConcreteProvider()
        result = provider.fetch_bars("EUR_USD", date(2024, 1, 1), date(2024, 1, 31))
        assert isinstance(result, pl.DataFrame)

    def test_fetch_bars_has_required_columns(self) -> None:
        provider = ConcreteProvider()
        result = provider.fetch_bars("EUR_USD", date(2024, 1, 1), date(2024, 1, 31))
        assert "timestamp" in result.columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_fetch_bars_validates_timeframe(self) -> None:
        provider = ConcreteProvider()
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            provider.fetch_bars(
                "EUR_USD", date(2024, 1, 1), date(2024, 1, 31), timeframe="invalid"
            )


class TestBaseProviderInstruments:
    """Tests for BaseProvider instrument methods."""

    def test_list_instruments_returns_dataframe(self) -> None:
        provider = ConcreteProvider()
        result = provider.list_instruments()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_list_instruments_with_asset_class_filter(self) -> None:
        provider = ConcreteProvider()
        result = provider.list_instruments(asset_class="forex")
        assert len(result) == 3

    def test_get_instrument_returns_dict(self) -> None:
        provider = ConcreteProvider()
        result = provider.get_instrument("EUR_USD")
        assert result is not None
        assert result["symbol"] == "EUR_USD"
        assert result["name"] == "Euro/USD"

    def test_get_instrument_returns_none_for_unknown(self) -> None:
        provider = ConcreteProvider()
        result = provider.get_instrument("UNKNOWN")
        assert result is None


class TestBaseProviderBarsToDataFrame:
    """Tests for BaseProvider.bars_to_dataframe static method."""

    def test_bars_to_dataframe_empty_list(self) -> None:
        result = BaseProvider.bars_to_dataframe([])
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert "timestamp" in result.columns
        assert "open" in result.columns

    def test_bars_to_dataframe_with_data(self, sample_bars_data: list[dict]) -> None:
        result = BaseProvider.bars_to_dataframe(sample_bars_data)
        assert len(result) == 3
        # Prices use Decimal for financial precision
        assert result["open"].dtype.is_decimal()
        assert result["volume"].dtype.is_decimal()

    def test_bars_to_dataframe_column_order(self, sample_bars_data: list[dict]) -> None:
        result = BaseProvider.bars_to_dataframe(sample_bars_data)
        expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        assert result.columns == expected_columns


class EmptyInstrumentsProvider(BaseProvider):
    """Provider with empty instruments for edge case testing."""

    @property
    def name(self) -> str:
        return "empty_provider"

    def fetch_bars(  # noqa: ARG002
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:  # noqa: ARG002
        return pl.DataFrame()


class TestBaseProviderEdgeCases:
    """Tests for BaseProvider edge cases."""

    def test_get_instrument_with_empty_instruments(self) -> None:
        provider = EmptyInstrumentsProvider()
        result = provider.get_instrument("EUR_USD")
        assert result is None


class NoSymbolColumnProvider(BaseProvider):
    """Provider with instruments but no symbol column."""

    @property
    def name(self) -> str:
        return "no_symbol_provider"

    def fetch_bars(  # noqa: ARG002
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:  # noqa: ARG002
        return pl.DataFrame({"name": ["Test"], "value": [1]})


class TestBaseProviderNoSymbolColumn:
    """Tests for provider with no symbol column."""

    def test_get_instrument_with_no_symbol_column(self) -> None:
        provider = NoSymbolColumnProvider()
        result = provider.get_instrument("EUR_USD")
        assert result is None

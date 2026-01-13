"""Tests for DataService programmatic API."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import logging
import polars as pl
import pytest
from liq.store.parquet import ParquetStore
from liq.store import key_builder

from liq.data.service import DataService
from liq.data.service import _timeframe_to_minutes
from liq.data.gaps import detect_gaps


def write_test_data(tmp_path: Path, provider: str, symbol: str, timeframe: str, df: pl.DataFrame) -> None:
    """Helper to write test data using ParquetStore."""
    store = ParquetStore(str(tmp_path))
    storage_key = f"{provider}/{key_builder.bars(symbol, timeframe)}"
    store.write(storage_key, df)


class TestDataServiceInit:
    """Tests for DataService initialization."""

    def test_init_with_defaults(self) -> None:
        """Test DataService initializes with default settings."""
        ds = DataService()

        assert ds is not None
        assert ds.settings is not None

    def test_init_with_custom_data_root(self, tmp_path: Path) -> None:
        """Test DataService initializes with custom data root."""
        ds = DataService(data_root=tmp_path)

        assert ds.data_root == tmp_path

    def test_init_creates_store(self, tmp_path: Path) -> None:
        """Test DataService creates a ParquetStore instance."""
        ds = DataService(data_root=tmp_path)

        assert ds.store is not None
        assert isinstance(ds.store, ParquetStore)


class TestDataServiceLoad:
    """Tests for DataService.load() method."""

    def test_load_returns_dataframe(self, tmp_path: Path) -> None:
        """Test load returns a Polars DataFrame."""
        # Setup: Create test data using ParquetStore
        test_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 10, 0, tzinfo=UTC)],
            "open": [1.0850],
            "high": [1.0875],
            "low": [1.0825],
            "close": [1.0860],
            "volume": [1000.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("oanda", "EUR_USD", "1m")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert "close" in result.columns

    def test_load_aggregates_from_1m(self, tmp_path: Path) -> None:
        """If higher timeframe missing, aggregate from 1m and persist."""
        test_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
            ],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10, 20, 30],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("oanda", "EUR_USD", "2m")
        assert len(result) == 2
        assert ds.store.exists(f"oanda/{key_builder.bars('EUR_USD', '2m')}")

    def test_load_refreshes_aggregate_when_1m_updates(self, tmp_path: Path, caplog) -> None:
        """Aggregates should refresh when base 1m data extends beyond cached range."""
        base_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            ],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [10, 20],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", base_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("oanda", "EUR_USD", "2m")
        assert result.height == 1

        new_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
            ],
            "open": [1.2, 1.3],
            "high": [1.3, 1.4],
            "low": [1.1, 1.2],
            "close": [1.25, 1.35],
            "volume": [30, 40],
        })
        ds.store.write(f"oanda/{key_builder.bars('EUR_USD', '1m')}", new_df, mode="append")

        with caplog.at_level(logging.INFO):
            refreshed = ds.load("oanda", "EUR_USD", "2m")
        assert refreshed.height == 2
        assert refreshed["timestamp"].max() == datetime(2024, 1, 1, 0, 2, tzinfo=UTC)
        assert "Refreshing cached 2m aggregate" in caplog.text

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test load raises FileNotFoundError for missing data."""
        ds = DataService(data_root=tmp_path)

        with pytest.raises(FileNotFoundError, match="Data not found"):
            ds.load("oanda", "MISSING_SYMBOL", "1m")


class TestDataServiceList:
    """Tests for DataService.list_symbols() method."""

    def test_list_symbols_empty(self, tmp_path: Path) -> None:
        """Test list_symbols returns empty list for empty data root."""
        ds = DataService(data_root=tmp_path)
        result = ds.list_symbols()

        assert result == []

    def test_list_symbols_with_data(self, tmp_path: Path) -> None:
        """Test list_symbols returns available data."""
        # Setup: Create test data using ParquetStore
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}))
        write_test_data(tmp_path, "binance", "BTC_USDT", "1h", pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}))

        ds = DataService(data_root=tmp_path)
        result = ds.list_symbols()

        assert len(result) == 2
        providers = [r["provider"] for r in result]
        assert "oanda" in providers
        assert "binance" in providers

    def test_list_symbols_ignores_non_bars(self, tmp_path: Path) -> None:
        """Non-bars keys should not appear in list_symbols output."""
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}))
        store = ParquetStore(str(tmp_path))
        store.write(
            "oanda/EUR_USD/features/test_v1",
            pl.DataFrame({"feature": [1.0]}),
            mode="overwrite",
        )
        store.write(
            "oanda/EUR_USD/quotes",
            pl.DataFrame({"bid": [1.0], "ask": [1.1]}),
            mode="overwrite",
        )

        ds = DataService(data_root=tmp_path)
        result = ds.list_symbols()

        assert len(result) == 1
        assert result[0]["symbol"] == "EUR_USD"

    def test_list_symbols_by_provider(self, tmp_path: Path) -> None:
        """Test list_symbols can filter by provider."""
        # Setup: Create test data using ParquetStore
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}))
        write_test_data(tmp_path, "binance", "BTC_USDT", "1m", pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}))

        ds = DataService(data_root=tmp_path)
        result = ds.list_symbols(provider="oanda")

        assert len(result) == 1
        assert result[0]["provider"] == "oanda"


class TestDataServiceFetch:
    """Tests for DataService.fetch() method."""

    def test_fetch_returns_dataframe(self, tmp_path: Path) -> None:
        """Test fetch returns data and stores to file."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
            "open": [1.0850],
            "high": [1.0875],
            "low": [1.0825],
            "close": [1.0860],
            "volume": [1000.0],
        })
        mock_provider.fetch_bars.return_value = mock_df

        ds = DataService(data_root=tmp_path)

        with patch.object(ds, "_get_provider", return_value=mock_provider):
            result = ds.fetch(
                provider="oanda",
                symbol="EUR_USD",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
            )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        mock_provider.fetch_bars.assert_called_once()

    def test_fetch_with_default_end_date(self, tmp_path: Path) -> None:
        """Test fetch uses today as default end date."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        mock_provider.fetch_bars.return_value = mock_df

        ds = DataService(data_root=tmp_path)

        with patch.object(ds, "_get_provider", return_value=mock_provider):
            ds.fetch(
                provider="oanda",
                symbol="EUR_USD",
                start=date(2024, 1, 15),
                # No end date provided - should default to today
                timeframe="1h",
                save=False,
            )

        # Verify fetch_bars was called with today's date as end
        call_args = mock_provider.fetch_bars.call_args
        assert call_args[0][2] == date.today()

    def test_fetch_saves_to_store(self, tmp_path: Path) -> None:
        """Test fetch saves data to ParquetStore."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
            "open": [1.0850],
            "high": [1.0875],
            "low": [1.0825],
            "close": [1.0860],
            "volume": [1000.0],
        })
        mock_provider.fetch_bars.return_value = mock_df

        ds = DataService(data_root=tmp_path)

        with patch.object(ds, "_get_provider", return_value=mock_provider):
            ds.fetch(
                provider="oanda",
                symbol="EUR_USD",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
                save=True,
            )

        # Verify data was saved via store
        assert ds.store.exists(f"oanda/{key_builder.bars('EUR_USD', '1h')}")
        loaded = ds.load("oanda", "EUR_USD", "1h")
        assert len(loaded) == 1

    def test_fetch_no_save_option(self, tmp_path: Path) -> None:
        """Test fetch can skip saving to store."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        mock_provider.fetch_bars.return_value = mock_df

        ds = DataService(data_root=tmp_path)

        with patch.object(ds, "_get_provider", return_value=mock_provider):
            result = ds.fetch(
                provider="oanda",
                symbol="EUR_USD",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
                save=False,
            )

        assert isinstance(result, pl.DataFrame)
        assert not ds.store.exists("oanda/EUR_USD/1h")


class TestDataServiceValidate:
    """Tests for DataService.validate() method."""

    def test_validate_returns_validation_result(self, tmp_path: Path) -> None:
        """Test validate returns validation result dict."""
        # Setup: Create test data using ParquetStore
        test_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 1, tzinfo=UTC),
            ],
            "open": [1.0850, 1.0860],
            "high": [1.0875, 1.0880],
            "low": [1.0825, 1.0830],
            "close": [1.0860, 1.0870],
            "volume": [1000.0, 1100.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.validate("oanda", "EUR_USD", "1m")

        assert isinstance(result, dict)
        assert "valid" in result
        assert "row_count" in result

    def test_validate_detects_nulls(self, tmp_path: Path) -> None:
        """Test validate detects null values."""
        test_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 10, 0, tzinfo=UTC)],
            "open": [1.0850],
            "high": [None],  # Null value
            "low": [1.0825],
            "close": [1.0860],
            "volume": [1000.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.validate("oanda", "EUR_USD", "1m")

        assert result["null_count"] > 0

    def test_validate_detects_duplicates(self, tmp_path: Path) -> None:
        """Test validate detects duplicate timestamps.

        Note: ParquetStore deduplicates by timestamp on write,
        so we write two separate DataFrames to create duplicates in memory.
        """
        # First write some data
        ts = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        test_df = pl.DataFrame({
            "timestamp": [ts, ts],  # Duplicate timestamps
            "open": [1.0850, 1.0860],
            "high": [1.0875, 1.0880],
            "low": [1.0825, 1.0830],
            "close": [1.0860, 1.0870],
            "volume": [1000.0, 1100.0],
        })
        # Write directly to test duplicate detection (store dedupes, so use raw write)
        data_dir = tmp_path / "oanda" / "EUR_USD" / "bars" / "1m"
        data_dir.mkdir(parents=True)
        test_df.write_parquet(data_dir / "data.parquet")

        ds = DataService(data_root=tmp_path)
        result = ds.validate("oanda", "EUR_USD", "1m")

        assert result["valid"] is False
        assert any("duplicate" in e for e in result["errors"])

    def test_gaps_detection(self, tmp_path: Path) -> None:
        """Test gaps method returns missing intervals."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 2, tzinfo=UTC),  # gap at 0:01
            ],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", df)

        ds = DataService(data_root=tmp_path)
        gaps = ds.gaps("oanda", "EUR_USD", "1m", expected_minutes=1)
        assert gaps  # gap detected

    def test_backfill_fetches_missing(self, tmp_path: Path) -> None:
        """Backfill should fetch missing ranges only."""
        # existing bar
        existing = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", existing)

        fetched = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 0, 1, tzinfo=UTC)],
            "open": [1.1],
            "high": [1.1],
            "low": [1.1],
            "close": [1.1],
            "volume": [1.0],
        })

        ds = DataService(data_root=tmp_path)
        mock_provider = MagicMock()
        mock_provider.fetch_bars.return_value = fetched
        with patch.object(ds, "_get_provider", return_value=mock_provider):
            combined = ds.backfill("oanda", "EUR_USD", start=date(2024, 1, 1), end=date(2024, 1, 1), timeframe="1m")

        assert combined.height == 2
        assert ds.store.exists(f"oanda/{key_builder.bars('EUR_USD', '1m')}")
        assert mock_provider.fetch_bars.called


class TestDataServiceInfo:
    """Tests for DataService.info() method."""

    def test_info_returns_metadata(self, tmp_path: Path) -> None:
        """Test info returns data metadata."""
        # Setup: Create test data using ParquetStore
        test_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 1, tzinfo=UTC),
            ],
            "open": [1.0850, 1.0860],
            "high": [1.0875, 1.0880],
            "low": [1.0825, 1.0830],
            "close": [1.0860, 1.0870],
            "volume": [1000.0, 1100.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.info("oanda", "EUR_USD", "1m")

        assert isinstance(result, dict)
        assert "row_count" in result
        assert result["row_count"] == 2
        assert "columns" in result
        assert "start" in result
        assert "end" in result


class TestDataServiceStats:
    """Tests for DataService.stats() method."""

    def test_stats_returns_statistics(self, tmp_path: Path) -> None:
        """Test stats returns statistical summary."""
        # Setup: Create test data using ParquetStore
        test_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 1, tzinfo=UTC),
                datetime(2024, 1, 15, 10, 2, tzinfo=UTC),
            ],
            "open": [1.0850, 1.0860, 1.0870],
            "high": [1.0875, 1.0880, 1.0890],
            "low": [1.0825, 1.0830, 1.0840],
            "close": [1.0860, 1.0870, 1.0880],
            "volume": [1000.0, 1100.0, 1200.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.stats("oanda", "EUR_USD", "1m")

        assert isinstance(result, dict)
        assert "close" in result
        assert "mean" in result["close"]
        assert "min" in result["close"]
        assert "max" in result["close"]


class TestDataServiceDelete:
    """Tests for DataService.delete() method."""

    def test_delete_removes_data(self, tmp_path: Path) -> None:
        """Test delete removes data from store."""
        # Setup: Create test data using ParquetStore
        test_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            "value": [1.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        assert ds.exists("oanda", "EUR_USD", "1m")

        result = ds.delete("oanda", "EUR_USD", "1m")

        assert result is True
        assert not ds.exists("oanda", "EUR_USD", "1m")

    def test_delete_returns_false_for_missing(self, tmp_path: Path) -> None:
        """Test delete returns False for missing data."""
        ds = DataService(data_root=tmp_path)
        result = ds.delete("oanda", "MISSING", "1m")

        assert result is False


class TestDataServiceExists:
    """Tests for DataService.exists() method."""

    def test_exists_returns_true_for_existing(self, tmp_path: Path) -> None:
        """Test exists returns True when data exists."""
        test_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            "value": [1.0],
        })
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        assert ds.exists("oanda", "EUR_USD", "1m") is True

    def test_exists_returns_false_for_missing(self, tmp_path: Path) -> None:
        """Test exists returns False when data doesn't exist."""
        ds = DataService(data_root=tmp_path)
        assert ds.exists("oanda", "MISSING", "1m") is False


class TestDataServiceGetProvider:
    """Tests for DataService._get_provider() method."""

    def test_get_provider_oanda(self, tmp_path: Path) -> None:
        """Test _get_provider returns OandaProvider for 'oanda'."""
        ds = DataService(data_root=tmp_path)

        mock_provider = MagicMock()
        mock_factory = MagicMock(return_value=mock_provider)

        # Patch the class variable directly
        with patch.dict(DataService._PROVIDER_FACTORIES, {"oanda": mock_factory}):
            provider = ds._get_provider("oanda")

            mock_factory.assert_called_once_with(ds.settings)
            assert provider is mock_provider

    def test_get_provider_binance(self, tmp_path: Path) -> None:
        """Test _get_provider returns BinanceProvider for 'binance'."""
        ds = DataService(data_root=tmp_path)

        mock_provider = MagicMock()
        mock_factory = MagicMock(return_value=mock_provider)

        # Patch the class variable directly
        with patch.dict(DataService._PROVIDER_FACTORIES, {"binance": mock_factory}):
            provider = ds._get_provider("binance")

            mock_factory.assert_called_once_with(ds.settings)
            assert provider is mock_provider

    def test_get_provider_unknown_raises(self, tmp_path: Path) -> None:
        """Test _get_provider raises for unknown provider."""
        ds = DataService(data_root=tmp_path)

        with pytest.raises(ValueError, match="Unknown provider"):
            ds._get_provider("unknown_provider")


class TestDataServiceImport:
    """Tests for DataService import from package."""

    def test_import_from_package(self) -> None:
        """Test DataService can be imported from liq.data."""
        from liq.data import DataService as DS

        assert DS is DataService


class TestDataServiceExtended:
    """Additional coverage for optional DataService helpers."""

    def test_fetch_quotes_saved(self, tmp_path: Path) -> None:
        """Quotes fetch should persist when supported."""
        ds = DataService(data_root=tmp_path)
        quotes = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            "bid": [1.0],
            "ask": [1.1],
        })

        mock_provider = MagicMock()
        mock_provider.fetch_quotes.return_value = quotes

        with patch.object(ds, "_get_provider", return_value=mock_provider):
            result = ds.fetch_quotes("oanda", "EUR_USD", date(2024, 1, 1), save=True, mode="overwrite")

        assert result.rows() == quotes.rows()
        assert ds.store.exists("oanda/EUR_USD/quotes")

    def test_fetch_quotes_unsupported(self, tmp_path: Path) -> None:
        """Quotes fetch should raise when provider lacks support."""
        ds = DataService(data_root=tmp_path)

        class NoQuotes: ...

        with patch.object(ds, "_get_provider", return_value=NoQuotes()):
            with pytest.raises(ValueError):
                ds.fetch_quotes("oanda", "EUR_USD", date(2024, 1, 1))

    def test_fetch_fundamentals_and_corp_actions(self, tmp_path: Path) -> None:
        """Fundamentals and corporate actions branches are covered."""
        ds = DataService(data_root=tmp_path)
        fundamentals = {"pe": 10}
        corp_actions = [{"type": "dividend", "amount": 1.0}]

        class Provider:
            def fetch_fundamentals(self, symbol: str, as_of: date) -> dict[str, float]:
                return fundamentals

            def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[dict[str, float]]:
                return corp_actions

        provider = Provider()

        with patch.object(ds, "_get_provider", return_value=provider):
            fetched_fundamentals = ds.fetch_fundamentals("oanda", "EUR_USD", date(2024, 1, 1), save=True)
            fetched_actions = ds.fetch_corporate_actions(
                "oanda", "EUR_USD", start=date(2024, 1, 1), end=date(2024, 1, 2), save=True
            )

        assert fetched_fundamentals == fundamentals
        assert fetched_actions == corp_actions
        assert ds.store.exists("oanda/EUR_USD/fundamentals")
        assert ds.store.exists("oanda/EUR_USD/corp_actions")

    def test_fetch_instruments_and_universe(self, tmp_path: Path) -> None:
        """Instrument and universe helpers delegate to provider."""
        ds = DataService(data_root=tmp_path)

        class Provider:
            def fetch_instruments(self, asset_class: str) -> list[str]:
                return ["EUR_USD"]

            def get_universe(self, asset_class: str, as_of: date | None = None) -> list[str]:
                return ["EUR_USD", "GBP_USD"]

            def validate_credentials(self) -> bool:
                return True

        provider = Provider()

        with patch.object(ds, "_get_provider", return_value=provider):
            instruments = ds.fetch_instruments("oanda", "forex")
            universe = ds.get_universe("oanda", "forex", as_of=date(2024, 1, 1))
            creds_ok = ds.validate_credentials("oanda")

        assert instruments == ["EUR_USD"]
        assert "GBP_USD" in universe
        assert creds_ok is True

    def test_list_symbols_with_bars_prefix(self, tmp_path: Path) -> None:
        """list_symbols should parse provider-prefixed bar keys."""
        store = ParquetStore(str(tmp_path))
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]})
        store.write(f"oanda/{key_builder.bars('EUR_USD', '1m')}", df)

        ds = DataService(data_root=tmp_path)
        symbols = ds.list_symbols()

        assert {"provider": "oanda", "symbol": "EUR_USD", "timeframe": "1m"} in symbols

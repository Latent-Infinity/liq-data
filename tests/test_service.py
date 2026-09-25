"""Tests for DataService programmatic API."""

import json
import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from liq.data.exceptions import (
    LockboxViolationError,
    SchemaValidationError,
    StorageVenueMismatchError,
    UndeclaredReadError,
)
from liq.data.providers.binance import BinanceProvider
from liq.data.service import BarCoverageObservation, DataService
from liq.data.settings import LiqDataSettings
from liq.data.universes import UniverseDefinition, UniverseKind
from liq.store import key_builder
from liq.store.parquet import ParquetStore


#: Declaration for reads made by this suite's own fixtures. They are not
#: research, so they declare ``dev_smoke`` and name the component in ``arm_id``
#: rather than minting a purpose token for test scaffolding.
class _SmokeRead(TypedDict):
    purpose: str
    arm_id: str


SMOKE: _SmokeRead = {"purpose": "dev_smoke", "arm_id": "liq_data_service_tests"}


def write_test_data(
    tmp_path: Path, provider: str, symbol: str, timeframe: str, df: pl.DataFrame
) -> None:
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
        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15, 10, 0, tzinfo=UTC)],
                "open": [1.0850],
                "high": [1.0875],
                "low": [1.0825],
                "close": [1.0860],
                "volume": [1000.0],
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("oanda", "EUR_USD", "1m", **SMOKE)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        assert "close" in result.columns

    def test_load_aggregates_from_1m(self, tmp_path: Path) -> None:
        """If higher timeframe missing, aggregate from 1m and persist."""
        test_df = pl.DataFrame(
            {
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
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("oanda", "EUR_USD", "2m", **SMOKE)
        assert len(result) == 2
        assert ds.store.exists(f"oanda/{key_builder.bars('EUR_USD', '2m')}")

    def test_load_refreshes_aggregate_when_1m_updates(self, tmp_path: Path, caplog) -> None:
        """Aggregates should refresh when base 1m data extends beyond cached range."""
        base_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                ],
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.05, 1.15],
                "volume": [10, 20],
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", base_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("oanda", "EUR_USD", "2m", **SMOKE)
        assert result.height == 1

        new_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
                ],
                "open": [1.2, 1.3],
                "high": [1.3, 1.4],
                "low": [1.1, 1.2],
                "close": [1.25, 1.35],
                "volume": [30, 40],
            }
        )
        ds.store.write(f"oanda/{key_builder.bars('EUR_USD', '1m')}", new_df, mode="append")

        with caplog.at_level(logging.INFO):
            refreshed = ds.load("oanda", "EUR_USD", "2m", **SMOKE)
        assert refreshed.height == 2
        assert refreshed["timestamp"].max() == datetime(2024, 1, 1, 0, 2, tzinfo=UTC)
        assert "Refreshing cached 2m aggregate" in caplog.text

    def test_load_refresh_keeps_daily_history_predating_1m_base(self, tmp_path: Path) -> None:
        """Refreshing a daily aggregate must not drop history older than the 1m base."""
        daily_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2017, 1, 3, 21, 0, tzinfo=UTC),
                    datetime(2018, 1, 2, 21, 0, tzinfo=UTC),
                    datetime(2019, 1, 2, 21, 0, tzinfo=UTC),
                    datetime(2020, 1, 2, 21, 0, tzinfo=UTC),
                ],
                "open": [1.0, 2.0, 3.0, 4.0],
                "high": [1.5, 2.5, 3.5, 4.5],
                "low": [0.5, 1.5, 2.5, 3.5],
                "close": [1.2, 2.2, 3.2, 4.2],
                "volume": [10.0, 20.0, 30.0, 40.0],
            }
        )
        write_test_data(tmp_path, "tradestation", "SPY", "1d", daily_df)

        base_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 2, 14, 30, tzinfo=UTC),
                    datetime(2020, 1, 2, 14, 31, tzinfo=UTC),
                    datetime(2020, 1, 3, 14, 30, tzinfo=UTC),
                ],
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.2, 11.2, 12.2],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        write_test_data(tmp_path, "tradestation", "SPY", "1m", base_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("tradestation", "SPY", "1d", **SMOKE)

        timestamps = result["timestamp"].to_list()
        # History older than the 1m base survives the rebuild.
        assert datetime(2017, 1, 3, 21, 0, tzinfo=UTC) in timestamps
        assert datetime(2018, 1, 2, 21, 0, tzinfo=UTC) in timestamps
        assert datetime(2019, 1, 2, 21, 0, tzinfo=UTC) in timestamps
        # The covered span is rebuilt from 1m: one bar per covered day, refreshed values.
        covered = result.filter(pl.col("timestamp") >= datetime(2020, 1, 1, tzinfo=UTC))
        assert covered.height == 2
        assert covered["close"].to_list() == [11.2, 12.2]
        assert result.height == 5

    def test_load_refresh_keeps_hourly_history_predating_1m_base(self, tmp_path: Path) -> None:
        """The same preservation applies to other aggregated timeframes."""
        hourly_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2023, 12, 29, 15, 0, tzinfo=UTC),
                    datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
                ],
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            }
        )
        write_test_data(tmp_path, "tradestation", "QQQ", "1h", hourly_df)

        base_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
                    datetime(2024, 1, 2, 15, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 16, 0, tzinfo=UTC),
                ],
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.2, 11.2, 12.2],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        write_test_data(tmp_path, "tradestation", "QQQ", "1m", base_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("tradestation", "QQQ", "1h", **SMOKE)

        timestamps = result["timestamp"].to_list()
        assert datetime(2023, 12, 29, 15, 0, tzinfo=UTC) in timestamps
        assert timestamps == [
            datetime(2023, 12, 29, 15, 0, tzinfo=UTC),
            datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
            datetime(2024, 1, 2, 16, 0, tzinfo=UTC),
        ]
        # The cached in-span bar is rebuilt from 1m rather than kept alongside it.
        assert result["close"].to_list() == [1.2, 11.2, 12.2]

    def test_load_refresh_keeps_history_with_decimal_bar_columns(self, tmp_path: Path) -> None:
        """History preservation holds for the decimal OHLCV dtypes stores use."""
        decimal_schema = {name: pl.Decimal(38, 8) for name in ("open", "high", "low", "close")}
        decimal_schema["volume"] = pl.Decimal(38, 2)

        def bars(rows: list[tuple[datetime, str, str]]) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "timestamp": [row[0] for row in rows],
                    "open": [Decimal(row[1]) for row in rows],
                    "high": [Decimal(row[1]) for row in rows],
                    "low": [Decimal(row[1]) for row in rows],
                    "close": [Decimal(row[1]) for row in rows],
                    "volume": [Decimal(row[2]) for row in rows],
                },
                schema_overrides=decimal_schema,
            )

        write_test_data(
            tmp_path,
            "tradestation",
            "DIA",
            "1d",
            bars(
                [
                    (datetime(2017, 1, 3, 21, 0, tzinfo=UTC), "100.00000000", "1000.00"),
                    (datetime(2020, 1, 2, 21, 0, tzinfo=UTC), "300.00000000", "3000.00"),
                ]
            ),
        )
        write_test_data(
            tmp_path,
            "tradestation",
            "DIA",
            "1m",
            bars(
                [
                    (datetime(2020, 1, 2, 14, 30, tzinfo=UTC), "310.00000000", "10.00"),
                    (datetime(2020, 1, 3, 14, 30, tzinfo=UTC), "320.00000000", "20.00"),
                ]
            ),
        )

        ds = DataService(data_root=tmp_path)
        result = ds.load("tradestation", "DIA", "1d", **SMOKE)

        assert result["timestamp"].to_list() == [
            datetime(2017, 1, 3, 21, 0, tzinfo=UTC),
            datetime(2020, 1, 2, tzinfo=UTC),
            datetime(2020, 1, 3, tzinfo=UTC),
        ]
        assert result["close"].to_list() == [
            Decimal("100.00000000"),
            Decimal("310.00000000"),
            Decimal("320.00000000"),
        ]

    def test_load_refresh_unchanged_when_1m_base_covers_cached_span(self, tmp_path: Path) -> None:
        """When the 1m base covers the cached span, the rebuild replaces it wholesale."""
        daily_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
                "open": [99.0],
                "high": [99.0],
                "low": [99.0],
                "close": [99.0],
                "volume": [1.0],
            }
        )
        write_test_data(tmp_path, "tradestation", "IWM", "1d", daily_df)

        base_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 14, 30, tzinfo=UTC),
                    datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
                ],
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            }
        )
        write_test_data(tmp_path, "tradestation", "IWM", "1m", base_df)

        ds = DataService(data_root=tmp_path)
        result = ds.load("tradestation", "IWM", "1d", **SMOKE)

        assert result["timestamp"].to_list() == [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        ]
        assert result["close"].to_list() == [1.2, 2.2]

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test load raises FileNotFoundError for missing data."""
        ds = DataService(data_root=tmp_path)

        with pytest.raises(FileNotFoundError, match="Data not found"):
            ds.load("oanda", "MISSING_SYMBOL", "1m", **SMOKE)


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
        write_test_data(
            tmp_path,
            "oanda",
            "EUR_USD",
            "1m",
            pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}),
        )
        write_test_data(
            tmp_path,
            "binance",
            "BTC_USDT",
            "1h",
            pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}),
        )

        ds = DataService(data_root=tmp_path)
        result = ds.list_symbols()

        assert len(result) == 2
        providers = [r["provider"] for r in result]
        assert "oanda" in providers
        assert "binance" in providers

    def test_list_symbols_ignores_non_bars(self, tmp_path: Path) -> None:
        """Non-bars keys should not appear in list_symbols output."""
        write_test_data(
            tmp_path,
            "oanda",
            "EUR_USD",
            "1m",
            pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}),
        )
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
        write_test_data(
            tmp_path,
            "oanda",
            "EUR_USD",
            "1m",
            pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}),
        )
        write_test_data(
            tmp_path,
            "binance",
            "BTC_USDT",
            "1m",
            pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "value": [1.0]}),
        )

        ds = DataService(data_root=tmp_path)
        result = ds.list_symbols(provider="oanda")

        assert len(result) == 1
        assert result[0]["provider"] == "oanda"


class TestDataServiceFetch:
    """Tests for DataService.fetch() method."""

    def test_fetch_returns_dataframe(self, tmp_path: Path) -> None:
        """Test fetch returns data and stores to file."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
                "open": [1.0850],
                "high": [1.0875],
                "low": [1.0825],
                "close": [1.0860],
                "volume": [1000.0],
            }
        )
        mock_provider.fetch_bars.return_value = mock_df

        ds = DataService(data_root=tmp_path)

        with patch.object(ds, "_get_provider", return_value=mock_provider):
            result = ds.fetch(
                provider="oanda",
                symbol="EUR_USD",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
                **SMOKE,
            )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1
        mock_provider.fetch_bars.assert_called_once()

    def test_fetch_with_default_end_date(self, tmp_path: Path) -> None:
        """Test fetch uses today as default end date."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [100.0],
            }
        )
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
                **SMOKE,
            )

        # Verify fetch_bars was called with today's date as end
        call_args = mock_provider.fetch_bars.call_args
        assert call_args[0][2] == date.today()

    def test_fetch_saves_to_store(self, tmp_path: Path) -> None:
        """Test fetch saves data to ParquetStore."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
                "open": [1.0850],
                "high": [1.0875],
                "low": [1.0825],
                "close": [1.0860],
                "volume": [1000.0],
            }
        )
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
                **SMOKE,
            )

        # Verify data was saved via store
        assert ds.store.exists(f"oanda/{key_builder.bars('EUR_USD', '1h')}")
        loaded = ds.load("oanda", "EUR_USD", "1h", **SMOKE)
        assert len(loaded) == 1

    def test_fetch_no_save_option(self, tmp_path: Path) -> None:
        """Test fetch can skip saving to store."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [100.0],
            }
        )
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
                **SMOKE,
            )

        assert isinstance(result, pl.DataFrame)
        assert not ds.store.exists("oanda/EUR_USD/1h")

    def test_declared_research_fetch_routes_through_guard(self, tmp_path: Path) -> None:
        """Provider reads need the same lockbox boundary as stored-data reads."""
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(ds, "_guard_research_read") as guard,
            patch.object(ds, "_get_provider") as get_provider,
        ):
            get_provider.return_value.fetch_bars.return_value = pl.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.05],
                    "volume": [100.0],
                }
            )
            ds.fetch(
                "tradestation",
                "@MBT",
                date(2024, 1, 15),
                date(2024, 1, 15),
                "1d",
                save=False,
                purpose="dev_smoke",
                arm_id="path_e_crypto_basis_carry",
                asset_class="future",
            )

        guard.assert_called_once_with(
            "tradestation",
            "@MBT",
            date(2024, 1, 15),
            date(2024, 1, 15),
            purpose="dev_smoke",
            arm_id="path_e_crypto_basis_carry",
            final_portfolio_review=False,
            asset_class="future",
            timeframe="1d",
        )

    def test_guard_rejects_fetch_before_provider_call(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(ds, "_guard_research_read", side_effect=LockboxViolationError("blocked")),
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(LockboxViolationError, match="blocked"),
        ):
            ds.fetch(
                "tradestation",
                "@MBT",
                date(2026, 1, 1),
                date(2026, 1, 2),
                "1d",
                save=False,
                purpose="discovery",
                arm_id="path_e_crypto_basis_carry",
                asset_class="future",
            )

        get_provider.assert_not_called()


class TestDataServiceValidate:
    """Tests for DataService.validate() method."""

    def test_validate_returns_validation_result(self, tmp_path: Path) -> None:
        """Test validate returns validation result dict."""
        # Setup: Create test data using ParquetStore
        test_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                    datetime(2024, 1, 15, 10, 1, tzinfo=UTC),
                ],
                "open": [1.0850, 1.0860],
                "high": [1.0875, 1.0880],
                "low": [1.0825, 1.0830],
                "close": [1.0860, 1.0870],
                "volume": [1000.0, 1100.0],
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.validate("oanda", "EUR_USD", "1m", **SMOKE)

        assert isinstance(result, dict)
        assert "valid" in result
        assert "row_count" in result

    def test_validate_detects_nulls(self, tmp_path: Path) -> None:
        """Test validate detects null values."""
        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 15, 10, 0, tzinfo=UTC)],
                "open": [1.0850],
                "high": [None],  # Null value
                "low": [1.0825],
                "close": [1.0860],
                "volume": [1000.0],
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.validate("oanda", "EUR_USD", "1m", **SMOKE)

        assert result["null_count"] > 0

    def test_validate_detects_duplicates(self, tmp_path: Path) -> None:
        """Test validate detects duplicate timestamps.

        Note: ParquetStore deduplicates by timestamp on write,
        so we write two separate DataFrames to create duplicates in memory.
        """
        # First write some data
        ts = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        test_df = pl.DataFrame(
            {
                "timestamp": [ts, ts],  # Duplicate timestamps
                "open": [1.0850, 1.0860],
                "high": [1.0875, 1.0880],
                "low": [1.0825, 1.0830],
                "close": [1.0860, 1.0870],
                "volume": [1000.0, 1100.0],
            }
        )
        # Write directly to test duplicate detection (store dedupes, so use raw write)
        data_dir = tmp_path / "oanda" / "EUR_USD" / "bars" / "1m"
        data_dir.mkdir(parents=True)
        test_df.write_parquet(data_dir / "data.parquet")

        ds = DataService(data_root=tmp_path)
        result = ds.validate("oanda", "EUR_USD", "1m", **SMOKE)

        assert result["valid"] is False
        assert any("duplicate" in e for e in result["errors"])

    def test_gaps_detection(self, tmp_path: Path) -> None:
        """Test gaps method returns missing intervals."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 0, 2, tzinfo=UTC),  # gap at 0:01
                ],
                "open": [1.0, 1.0],
                "high": [1.0, 1.0],
                "low": [1.0, 1.0],
                "close": [1.0, 1.0],
                "volume": [1.0, 1.0],
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", df)

        ds = DataService(data_root=tmp_path)
        gaps = ds.gaps("oanda", "EUR_USD", "1m", expected_minutes=1, **SMOKE)
        assert gaps  # gap detected

    def test_backfill_fetches_missing(self, tmp_path: Path) -> None:
        """Backfill should fetch missing ranges only."""
        # existing bar
        existing = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", existing)

        fetched = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 0, 1, tzinfo=UTC)],
                "open": [1.1],
                "high": [1.1],
                "low": [1.1],
                "close": [1.1],
                "volume": [1.0],
            }
        )

        ds = DataService(data_root=tmp_path)
        mock_provider = MagicMock()
        mock_provider.fetch_bars.return_value = fetched
        with patch.object(ds, "_get_provider", return_value=mock_provider):
            combined = ds.backfill(
                "oanda",
                "EUR_USD",
                start=date(2024, 1, 1),
                end=date(2024, 1, 1),
                timeframe="1m",
                **SMOKE,
            )

        assert combined.height == 2
        assert ds.store.exists(f"oanda/{key_builder.bars('EUR_USD', '1m')}")
        assert mock_provider.fetch_bars.called


class TestDataServiceInfo:
    """Tests for DataService.info() method."""

    def test_info_returns_metadata(self, tmp_path: Path) -> None:
        """Test info returns data metadata."""
        # Setup: Create test data using ParquetStore
        test_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 15, 10, 0, tzinfo=UTC),
                    datetime(2024, 1, 15, 10, 1, tzinfo=UTC),
                ],
                "open": [1.0850, 1.0860],
                "high": [1.0875, 1.0880],
                "low": [1.0825, 1.0830],
                "close": [1.0860, 1.0870],
                "volume": [1000.0, 1100.0],
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.info("oanda", "EUR_USD", "1m", **SMOKE)

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
        test_df = pl.DataFrame(
            {
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
            }
        )
        write_test_data(tmp_path, "oanda", "EUR_USD", "1m", test_df)

        ds = DataService(data_root=tmp_path)
        result = ds.stats("oanda", "EUR_USD", "1m", **SMOKE)

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
        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "value": [1.0],
            }
        )
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
        test_df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "value": [1.0],
            }
        )
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

    def test_get_provider_sec_edgar_uses_free_reference_factory(self, tmp_path: Path) -> None:
        settings = LiqDataSettings(
            data_root=tmp_path,
            sec_edgar_user_agent="test test@example.com",
        )

        provider = DataService(settings=settings)._get_corporate_actions_provider("sec_edgar")

        assert provider.name == "sec_edgar"

    def test_corporate_actions_provider_is_reused_for_a_service_session(
        self, tmp_path: Path
    ) -> None:
        ds = DataService(settings=LiqDataSettings(data_root=tmp_path))
        provider = MagicMock()
        factory = MagicMock(return_value=provider)

        with patch.dict(
            DataService._CORPORATE_ACTION_FACTORIES,
            {"sec_edgar": factory},
        ):
            first = ds._get_corporate_actions_provider("sec_edgar")
            second = ds._get_corporate_actions_provider("SEC_EDGAR")

        assert first is provider
        assert second is provider
        factory.assert_called_once_with(ds.settings)


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
        quotes = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "bid": [1.0],
                "ask": [1.1],
            }
        )

        mock_provider = MagicMock()
        mock_provider.fetch_quotes.return_value = quotes

        with patch.object(ds, "_get_provider", return_value=mock_provider):
            result = ds.fetch_quotes(
                "oanda", "EUR_USD", date(2024, 1, 1), save=True, mode="overwrite"
            )

        assert result.rows() == quotes.rows()
        assert ds.store.exists("oanda/EUR_USD/quotes")

    def test_fetch_quotes_unsupported(self, tmp_path: Path) -> None:
        """Quotes fetch should raise when provider lacks support."""
        ds = DataService(data_root=tmp_path)

        class NoQuotes: ...

        with (
            patch.object(ds, "_get_provider", return_value=NoQuotes()),
            pytest.raises(ValueError),
        ):
            ds.fetch_quotes("oanda", "EUR_USD", date(2024, 1, 1))

    def test_fetch_fundamentals_and_corp_actions(self, tmp_path: Path) -> None:
        """Fundamentals and corporate actions branches are covered."""
        ds = DataService(data_root=tmp_path)
        fundamentals = {"pe": 10}
        corp_actions = [{"type": "dividend", "amount": 1.0}]

        class Provider:
            def fetch_fundamentals(self, symbol: str, as_of: date) -> dict[str, float]:
                return fundamentals

            def get_corporate_actions(
                self, symbol: str, start: date, end: date
            ) -> list[dict[str, float]]:
                return corp_actions

        provider = Provider()

        with patch.object(ds, "_get_provider", return_value=provider):
            fetched_fundamentals = ds.fetch_fundamentals(
                "oanda", "EUR_USD", date(2024, 1, 1), save=True
            )
            fetched_actions = ds.fetch_corporate_actions(
                "oanda",
                "EUR_USD",
                start=date(2024, 1, 1),
                end=date(2024, 1, 2),
                save=True,
                **SMOKE,
            )

        assert fetched_fundamentals == fundamentals
        assert fetched_actions == corp_actions
        assert ds.store.exists("oanda/EUR_USD/fundamentals")
        assert ds.store.exists("oanda/EUR_USD/corp_actions")

    def test_declared_corporate_actions_fetch_routes_through_guard(self, tmp_path: Path) -> None:
        """Reference-data reads use the arm's frozen fold before provider access."""
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.get_corporate_actions.return_value = []

        with (
            patch.object(ds, "_guard_research_read") as guard,
            patch.object(ds, "_get_provider", return_value=provider),
        ):
            result = ds.fetch_corporate_actions(
                "alpaca",
                "AAPL",
                start=date(2017, 1, 1),
                end=date(2024, 12, 31),
                purpose="discovery",
                arm_id="index_reconstitution",
                asset_class="index_recon_book",
            )

        assert result == []
        guard.assert_called_once_with(
            "alpaca",
            "AAPL",
            date(2017, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="index_reconstitution",
            final_portfolio_review=False,
            asset_class="index_recon_book",
            data_kind="corporate_actions",
        )

    def test_corporate_actions_guard_rejects_before_provider_access(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(
                ds,
                "_guard_research_read",
                side_effect=LockboxViolationError("blocked"),
            ),
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(LockboxViolationError, match="blocked"),
        ):
            ds.fetch_corporate_actions(
                "alpaca",
                "AAPL",
                start=date(2026, 1, 1),
                end=date(2026, 1, 2),
                purpose="discovery",
                arm_id="index_reconstitution",
                asset_class="index_recon_book",
            )

        get_provider.assert_not_called()

    def test_sec_edgar_corporate_actions_use_index_recon_lockbox_before_provider_access(
        self, tmp_path: Path
    ) -> None:
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(LockboxViolationError, match="program lockbox"),
        ):
            ds.fetch_corporate_actions(
                "sec_edgar",
                "AAPL",
                start=date(2026, 1, 1),
                end=date(2026, 1, 2),
                purpose="discovery",
                arm_id="index_reconstitution",
                asset_class="index_recon_book",
            )

        get_provider.assert_not_called()

    def test_declared_adjustment_factors_fetch_routes_through_guard(self, tmp_path: Path) -> None:
        """Adjustment reference reads use the same frozen-fold guard."""
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.get_adjustment_factors.return_value = [{"price_factor": 0.5}]

        with (
            patch.object(ds, "_guard_research_read") as guard,
            patch.object(ds, "_get_provider", return_value=provider),
        ):
            result = ds.fetch_adjustment_factors(
                "databento",
                "AAPL",
                start=date(2018, 5, 1),
                end=date(2024, 12, 31),
                purpose="discovery",
                arm_id="index_reconstitution",
                asset_class="index_recon_book",
            )

        assert result == [{"price_factor": 0.5}]
        guard.assert_called_once_with(
            "databento",
            "AAPL",
            date(2018, 5, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="index_reconstitution",
            final_portfolio_review=False,
            asset_class="index_recon_book",
            data_kind="adjustment_factors",
        )
        provider.get_adjustment_factors.assert_called_once_with(
            "AAPL", date(2018, 5, 1), date(2024, 12, 31)
        )

    def test_adjustment_factors_guard_rejects_before_provider_access(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(
                ds,
                "_guard_research_read",
                side_effect=LockboxViolationError("blocked"),
            ),
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(LockboxViolationError, match="blocked"),
        ):
            ds.fetch_adjustment_factors(
                "databento",
                "AAPL",
                start=date(2026, 1, 1),
                end=date(2026, 1, 2),
                purpose="discovery",
                arm_id="index_reconstitution",
                asset_class="index_recon_book",
            )

        get_provider.assert_not_called()

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


class TestDataServiceBarCoverageProbe:
    """The Path D coverage API exposes timestamps, never price or volume values."""

    START = date(2021, 6, 15)
    END = date(2025, 12, 31)
    ARM = "path_d_short_interest"

    @staticmethod
    def _bars() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2021, 6, 15, tzinfo=UTC),
                    datetime(2021, 6, 15, tzinfo=UTC),
                    datetime(2025, 12, 31, tzinfo=UTC),
                    datetime(2026, 1, 2, tzinfo=UTC),
                ],
                "open": [10.0, 10.0, 20.0, 30.0],
                "high": [11.0, 11.0, 21.0, 31.0],
                "low": [9.0, 9.0, 19.0, 29.0],
                "close": [10.5, 10.5, 20.5, 30.5],
                "volume": [100.0, 100.0, 200.0, 300.0],
            }
        )

    def test_probe_returns_only_timestamp_coverage_metadata(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.fetch_bars.return_value = self._bars()

        with patch.object(ds, "_get_provider", return_value=provider):
            observation = ds.probe_bar_coverage(
                "tradestation",
                "OLD",
                self.START,
                self.END,
                timeframe="1d",
                purpose="characterization",
                arm_id=self.ARM,
                asset_class="short_interest_book",
            )

        assert isinstance(observation, BarCoverageObservation)
        assert observation.row_count == 4
        assert observation.unique_timestamp_count == 3
        assert observation.duplicate_timestamp_count == 1
        assert observation.outside_requested_window_count == 1
        assert observation.first_timestamp == "2021-06-15T00:00:00+00:00"
        assert observation.last_timestamp == "2026-01-02T00:00:00+00:00"
        payload = observation.to_dict()
        assert set(payload) == {
            "provider",
            "symbol",
            "timeframe",
            "requested_start",
            "requested_end",
            "row_count",
            "unique_timestamp_count",
            "duplicate_timestamp_count",
            "outside_requested_window_count",
            "first_timestamp",
            "last_timestamp",
            "timestamp_sha256",
        }
        assert not {"open", "high", "low", "close", "volume"} & payload.keys()
        provider.fetch_bars.assert_called_once_with("OLD", self.START, self.END, timeframe="1d")
        assert not list(tmp_path.rglob("*.parquet"))

        log = json.loads((tmp_path / "lockbox_usage_log.jsonl").read_text().strip())
        assert log["dataset"] == "short_interest_bar_coverage"
        assert log["purpose"] == "characterization"
        assert log["arm_id"] == self.ARM

    def test_probe_empty_response_is_explicit(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.fetch_bars.return_value = pl.DataFrame()

        with patch.object(ds, "_get_provider", return_value=provider):
            observation = ds.probe_bar_coverage(
                "tradestation",
                "GONE",
                self.START,
                self.END,
                timeframe="1d",
                purpose="characterization",
                arm_id=self.ARM,
                asset_class="short_interest_book",
            )

        assert observation.row_count == 0
        assert observation.first_timestamp is None
        assert observation.last_timestamp is None

    def test_probe_reuses_one_provider_session_across_symbols(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.fetch_bars.return_value = pl.DataFrame()

        with patch.object(ds, "_get_provider", return_value=provider) as get_provider:
            for symbol in ("GONE", "OLD"):
                ds.probe_bar_coverage(
                    "tradestation",
                    symbol,
                    self.START,
                    self.END,
                    timeframe="1d",
                    purpose="characterization",
                    arm_id=self.ARM,
                    asset_class="short_interest_book",
                )

        get_provider.assert_called_once_with("tradestation")
        assert provider.fetch_bars.call_count == 2

    @pytest.mark.parametrize(
        "frame",
        [
            pl.DataFrame({"close": [1.0]}),
            pl.DataFrame({"timestamp": [datetime(2021, 6, 15)]}),
        ],
    )
    def test_probe_rejects_missing_or_naive_timestamps(
        self, tmp_path: Path, frame: pl.DataFrame
    ) -> None:
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.fetch_bars.return_value = frame

        with (
            patch.object(ds, "_get_provider", return_value=provider),
            pytest.raises(SchemaValidationError),
        ):
            ds.probe_bar_coverage(
                "tradestation",
                "OLD",
                self.START,
                self.END,
                timeframe="1d",
                purpose="characterization",
                arm_id=self.ARM,
                asset_class="short_interest_book",
            )

    @pytest.mark.parametrize("operation", ["fetch", "load"])
    def test_ordinary_bar_read_is_rejected_before_data_access(
        self, tmp_path: Path, operation: str
    ) -> None:
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(LockboxViolationError, match="coverage probe only"),
        ):
            if operation == "fetch":
                ds.fetch(
                    "tradestation",
                    "OLD",
                    self.START,
                    self.END,
                    timeframe="1d",
                    save=False,
                    purpose="characterization",
                    arm_id=self.ARM,
                    asset_class="short_interest_book",
                )
            else:
                ds.load(
                    "tradestation",
                    "OLD",
                    "1d",
                    start=self.START,
                    end=self.END,
                    purpose="characterization",
                    arm_id=self.ARM,
                    asset_class="short_interest_book",
                )

        get_provider.assert_not_called()

    def test_short_interest_discriminator_cannot_omit_research_purpose(
        self, tmp_path: Path
    ) -> None:
        """Still refused, now under the type that says what is actually wrong.

        The omission used to surface as a ``LockboxViolationError`` naming the
        short-interest authorization, which reported a fold refusal for what
        was only a missing declaration. Default-deny refuses it earlier and
        types it as such; the discriminator's own fold rules are unchanged.
        """
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(UndeclaredReadError, match="declares no purpose"),
        ):
            ds.fetch(
                "tradestation",
                "OLD",
                self.START,
                self.END,
                timeframe="1d",
                save=False,
                asset_class="short_interest_book",
            )

        get_provider.assert_not_called()

    def test_probe_rejects_non_daily_timeframe_before_provider_access(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(LockboxViolationError, match="daily bars only"),
        ):
            ds.probe_bar_coverage(
                "tradestation",
                "OLD",
                self.START,
                self.END,
                timeframe="1m",
                purpose="characterization",
                arm_id=self.ARM,
                asset_class="short_interest_book",
            )

        get_provider.assert_not_called()

    def test_probe_rejects_unapproved_provider_before_provider_access(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)

        with (
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(LockboxViolationError, match="TradeStation only"),
        ):
            ds.probe_bar_coverage(
                "databento",
                "OLD",
                self.START,
                self.END,
                timeframe="1d",
                purpose="characterization",
                arm_id=self.ARM,
                asset_class="short_interest_book",
            )

        get_provider.assert_not_called()


class TestDataServiceResearchReadAssetClass:
    """asset_class must thread through the research-read guard so a futures read
    is fold-governed by ``tradestation_crypto_futures`` and not the equity cohort.

    Without threading, a declared-purpose read of ``@MBT``/``@MET`` silently
    resolves to ``tradestation_cohort_1m`` and borrows the equity cohort's fold
    windows -- a lockbox-governance bug.
    """

    ARM = "path_e_crypto_basis_carry"

    def test_future_read_governed_by_crypto_futures_windows(self, tmp_path: Path) -> None:
        """A future-class read resolves to the crypto-futures dataset: 2021
        precedes its frozen 2022 discovery start, so the read is rejected --
        the equity cohort (discovery from 2019) would have admitted it."""
        ds = DataService(data_root=tmp_path)
        with pytest.raises(LockboxViolationError):
            ds._guard_research_read(
                "tradestation",
                "@MBT",
                date(2021, 1, 1),
                date(2021, 12, 31),
                purpose="discovery",
                arm_id=self.ARM,
                final_portfolio_review=False,
                asset_class="future",
            )

    def test_unambiguous_future_symbol_is_safe_without_asset_class(self, tmp_path: Path) -> None:
        """Known ``@MBT``/``@MET`` roots cannot bypass their frozen fold by omission."""
        ds = DataService(data_root=tmp_path)
        with pytest.raises(LockboxViolationError):
            ds._guard_research_read(
                "tradestation",
                "@MBT",
                date(2021, 1, 1),
                date(2021, 12, 31),
                purpose="discovery",
                arm_id=self.ARM,
                final_portfolio_review=False,
            )

    def test_load_threads_asset_class_to_guard(self, tmp_path: Path) -> None:
        """End-to-end: ``load`` rejects an out-of-window futures discovery read
        via the guard before touching the store, proving ``asset_class`` threads
        through (2021 precedes the frozen 2022 crypto-futures discovery start)."""
        ds = DataService(data_root=tmp_path)
        with pytest.raises(LockboxViolationError):
            ds.load(
                "tradestation",
                "@MBT",
                "1d",
                date(2021, 1, 1),
                date(2021, 12, 31),
                purpose="discovery",
                arm_id=self.ARM,
                asset_class="future",
            )

    def test_undeclared_future_read_is_refused(self, tmp_path: Path) -> None:
        """purpose=None is refused regardless of asset_class.

        This is the inverted form of the former
        ``test_non_research_future_read_is_unguarded``: same call under test,
        opposite expectation. Omitting the declaration used to short-circuit
        the guard and return ``None``; it now raises, and the raised type says
        *the declaration is missing*, not *the fold forbids it*.
        """
        ds = DataService(data_root=tmp_path)
        with pytest.raises(UndeclaredReadError) as excinfo:
            ds._guard_research_read(
                "tradestation",
                "@MBT",
                None,
                None,
                purpose=None,
                arm_id=None,
                final_portfolio_review=False,
                asset_class="future",
            )
        assert not isinstance(excinfo.value, LockboxViolationError)


class TestUndeclaredReadsAreRefused:
    """Default-deny: an entry point that can return market values refuses a
    read that declares no ``purpose``, before any store or provider is touched.

    Every case below asserts refusal *and* non-vacuity: the mock standing in
    for the store or the provider records no call at all, so the refusal
    cannot be an artifact of the read having already happened.
    """

    ARM = "liq_data_default_deny_tests"
    START = date(2023, 1, 2)
    END = date(2023, 1, 31)

    @staticmethod
    def _bars(tmp_path: Path) -> None:
        write_test_data(
            tmp_path,
            "tradestation",
            "AAPL",
            "1m",
            pl.DataFrame(
                {
                    "timestamp": [
                        datetime(2023, 1, 3, 14, 30, tzinfo=UTC),
                        datetime(2023, 1, 3, 14, 31, tzinfo=UTC),
                    ],
                    "open": [1.0, 1.1],
                    "high": [1.2, 1.3],
                    "low": [0.9, 1.0],
                    "close": [1.1, 1.2],
                    "volume": [100, 110],
                }
            ),
        )

    def test_load_refuses_before_the_store_is_touched(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        with patch.object(ds, "_store") as store, pytest.raises(UndeclaredReadError):
            ds.load("tradestation", "AAPL", "1m", self.START, self.END)
        assert store.mock_calls == []

    def test_iter_batches_refuses_before_the_store_is_touched(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        with patch.object(ds, "_store") as store, pytest.raises(UndeclaredReadError):
            next(ds.iter_batches("tradestation", "AAPL", "1m", self.START, self.END))
        assert store.mock_calls == []

    def test_fetch_refuses_before_the_provider_is_touched(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        with (
            patch.object(ds, "_get_provider") as get_provider,
            patch.object(ds, "_store") as store,
            pytest.raises(UndeclaredReadError),
        ):
            ds.fetch("tradestation", "AAPL", self.START, self.END, timeframe="1m")
        get_provider.assert_not_called()
        assert store.mock_calls == []

    def test_fetch_corporate_actions_refuses_before_the_provider_is_touched(
        self, tmp_path: Path
    ) -> None:
        ds = DataService(data_root=tmp_path)
        with (
            patch.object(ds, "_get_corporate_actions_provider") as get_provider,
            pytest.raises(UndeclaredReadError),
        ):
            ds.fetch_corporate_actions("tradestation", "AAPL", self.START, self.END)
        get_provider.assert_not_called()

    def test_fetch_adjustment_factors_refuses_before_the_provider_is_touched(
        self, tmp_path: Path
    ) -> None:
        ds = DataService(data_root=tmp_path)
        with (
            patch.object(ds, "_get_provider") as get_provider,
            pytest.raises(UndeclaredReadError),
        ):
            ds.fetch_adjustment_factors("tradestation", "AAPL", self.START, self.END)
        get_provider.assert_not_called()

    def test_backfill_refuses_before_store_or_provider_is_touched(self, tmp_path: Path) -> None:
        """backfill returns market values and reads the store before it fetches,
        so its refusal has to precede both."""
        self._bars(tmp_path)
        ds = DataService(data_root=tmp_path)
        with (
            patch.object(ds, "_get_provider") as get_provider,
            patch.object(ds, "_store") as store,
            pytest.raises(UndeclaredReadError),
        ):
            ds.backfill("tradestation", "AAPL", self.START, self.END, "1m")
        get_provider.assert_not_called()
        assert store.mock_calls == []

    @pytest.mark.parametrize("method", ["validate", "info", "stats"])
    def test_derived_read_paths_refuse_before_the_store_is_touched(
        self, tmp_path: Path, method: str
    ) -> None:
        ds = DataService(data_root=tmp_path)
        with patch.object(ds, "_store") as store, pytest.raises(UndeclaredReadError):
            getattr(ds, method)("tradestation", "AAPL", "1m")
        assert store.mock_calls == []

    def test_gaps_refuses_before_the_store_is_touched(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        with patch.object(ds, "_store") as store, pytest.raises(UndeclaredReadError):
            ds.gaps("tradestation", "AAPL", "1m", 1)
        assert store.mock_calls == []

    def test_refusal_message_names_the_missing_declaration(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        with pytest.raises(UndeclaredReadError, match="declares no purpose"):
            ds.load("tradestation", "AAPL", "1m", self.START, self.END)


class TestUndeclaredReadIsNotAFoldViolation:
    """ "You didn't say" and "you may not" must be separable by type.

    A reader of a traceback, and an ``except`` clause, must be able to tell a
    missing declaration from a fold-boundary refusal without matching on
    message text.
    """

    ARM = "liq_data_default_deny_tests"

    def test_the_two_types_are_unrelated_by_inheritance(self) -> None:
        assert not issubclass(UndeclaredReadError, LockboxViolationError)
        assert not issubclass(LockboxViolationError, UndeclaredReadError)

    def test_catching_undeclared_does_not_catch_a_fold_violation(self, tmp_path: Path) -> None:
        """2021 precedes the frozen 2022 crypto-futures discovery start, so this
        declared read is a genuine fold violation and must escape a handler that
        only expects a missing declaration."""
        ds = DataService(data_root=tmp_path)
        with pytest.raises(LockboxViolationError) as excinfo:
            try:
                ds.load(
                    "tradestation",
                    "@MBT",
                    "1d",
                    date(2021, 1, 1),
                    date(2021, 12, 31),
                    purpose="discovery",
                    arm_id=self.ARM,
                    asset_class="future",
                )
            except UndeclaredReadError as wrong:  # pragma: no cover - must not fire
                raise AssertionError("fold violation was caught as an undeclared read") from wrong
        assert not isinstance(excinfo.value, UndeclaredReadError)

    def test_catching_fold_violations_does_not_catch_an_undeclared_read(
        self, tmp_path: Path
    ) -> None:
        ds = DataService(data_root=tmp_path)
        with pytest.raises(UndeclaredReadError) as excinfo:
            try:
                ds.load("tradestation", "@MBT", "1d", date(2021, 1, 1), date(2021, 12, 31))
            except LockboxViolationError as wrong:  # pragma: no cover - must not fire
                raise AssertionError("undeclared read was caught as a fold violation") from wrong
        assert not isinstance(excinfo.value, LockboxViolationError)


class TestDerivedReadPathsCanDeclare:
    """``validate``/``info``/``stats``/``gaps``/``backfill`` take and forward a
    declaration, and the declared read reaches the ledger."""

    ARM = "liq_data_service_diagnostics"

    @staticmethod
    def _bars(tmp_path: Path) -> None:
        write_test_data(
            tmp_path,
            "tradestation",
            "AAPL",
            "1m",
            pl.DataFrame(
                {
                    "timestamp": [
                        datetime(2023, 1, 3, 14, 30, tzinfo=UTC),
                        datetime(2023, 1, 3, 14, 31, tzinfo=UTC),
                    ],
                    "open": [1.0, 1.1],
                    "high": [1.2, 1.3],
                    "low": [0.9, 1.0],
                    "close": [1.1, 1.2],
                    "volume": [100, 110],
                }
            ),
        )

    @staticmethod
    def _log_entries(tmp_path: Path) -> list[dict]:
        log = tmp_path / "lockbox_usage_log.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    @pytest.mark.parametrize("method", ["validate", "info", "stats"])
    def test_declared_derived_read_is_ledgered(self, tmp_path: Path, method: str) -> None:
        self._bars(tmp_path)
        ds = DataService(data_root=tmp_path)
        getattr(ds, method)("tradestation", "AAPL", "1m", purpose="dev_smoke", arm_id=self.ARM)
        entries = self._log_entries(tmp_path)
        assert [e["arm_id"] for e in entries] == [self.ARM]
        assert entries[0]["purpose"] == "dev_smoke"
        assert entries[0]["dataset"] == "tradestation_cohort_1m"

    def test_declared_gaps_read_is_ledgered(self, tmp_path: Path) -> None:
        self._bars(tmp_path)
        ds = DataService(data_root=tmp_path)
        ds.gaps("tradestation", "AAPL", "1m", 1, purpose="dev_smoke", arm_id=self.ARM)
        entries = self._log_entries(tmp_path)
        assert [e["arm_id"] for e in entries] == [self.ARM]

    def test_declared_backfill_is_ledgered_and_bounded(self, tmp_path: Path) -> None:
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.fetch_bars.return_value = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 3, 14, 30, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.2],
                "low": [0.9],
                "close": [1.1],
                "volume": [100],
            }
        )
        with patch.object(ds, "_get_provider", return_value=provider):
            ds.backfill(
                "tradestation",
                "AAPL",
                date(2023, 1, 2),
                date(2023, 1, 31),
                "1m",
                purpose="dev_smoke",
                arm_id=self.ARM,
            )
        entries = self._log_entries(tmp_path)
        # Two entries: backfill guards its own window, then delegates the
        # empty-store case to ``fetch``, which guards again.
        assert {e["arm_id"] for e in entries} == {self.ARM}
        assert {(e["start"], e["end"]) for e in entries} == {("2023-01-02", "2023-01-31")}

    def test_declared_backfill_over_existing_bars_is_still_guarded(self, tmp_path: Path) -> None:
        """The gap-filling branch reads the provider directly rather than
        through ``fetch``; it must still be declared and ledgered."""
        self._bars(tmp_path)
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.fetch_bars.return_value = pl.DataFrame()
        with patch.object(ds, "_get_provider", return_value=provider):
            ds.backfill(
                "tradestation",
                "AAPL",
                date(2023, 1, 2),
                date(2023, 1, 31),
                "1m",
                purpose="dev_smoke",
                arm_id=self.ARM,
            )
        entries = self._log_entries(tmp_path)
        assert [e["arm_id"] for e in entries] == [self.ARM]

    def test_unbounded_derived_read_refuses_a_research_purpose(self, tmp_path: Path) -> None:
        """Condition 5: the unbounded relaxation reaches ``dev_smoke`` only."""
        self._bars(tmp_path)
        ds = DataService(data_root=tmp_path)
        with pytest.raises(ValueError, match="start and end"):
            ds.stats("tradestation", "AAPL", "1m", purpose="discovery", arm_id=self.ARM)


class _NamedProvider:
    """Minimal provider double that reports a venue the way a real one does."""

    def __init__(self, name: str, bars: pl.DataFrame | None = None) -> None:
        self._name = name
        self._bars = bars if bars is not None else _one_bar()
        self.fetch_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def fetch_bars(self, *args: object, **kwargs: object) -> pl.DataFrame:
        self.fetch_calls += 1
        return self._bars


def _one_bar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 15, tzinfo=UTC)],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41500.0],
            "close": [42200.0],
            "volume": [100.0],
        }
    )


class TestOneVenueOneStorageKey:
    """A bar is reachable under exactly one key, and that key names its venue.

    The defect this closes was documented in this workspace in February 2026 and
    shipped unrouted: ``DataService`` builds the storage key from the caller's
    provider string, never from the resolved provider's ``name``, so the venue a
    fetch contacted reached no storage decision at all. Stored bars carry no
    venue provenance, so nothing downstream could ever detect the mislabel.
    """

    SMOKE: _SmokeRead = {"purpose": "dev_smoke", "arm_id": "liq_data_venue_tests"}

    def test_binance_key_admits_a_binance_us_fetch(self, tmp_path: Path) -> None:
        """The measured truth: this key's tree holds Binance.US bars.

        Under ``BINANCE_USE_US=true`` the provider reports ``binance_us`` while
        the caller writes under ``binance``. That used to be a silent mislabel.
        It is now the key's recorded venue, so the write is truthful and the
        venue stays in one key rather than being split across two.
        """
        ds = DataService(data_root=tmp_path)
        provider = _NamedProvider("binance_us")

        with patch.object(ds, "_get_provider", return_value=provider):
            ds.fetch(
                "binance", "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), "1m", **self.SMOKE
            )

        assert provider.fetch_calls == 1
        assert ds.store.exists(f"binance/{key_builder.bars('BTC_USDT', '1m')}")

    def test_binance_key_refuses_a_binance_com_fetch(self, tmp_path: Path) -> None:
        """No market mixing: the plan of record forbids it for this surface.

        ``api.binance.com`` returns HTTP 451 from this environment today, which
        is a location fact and not a control. If it ever lifts, a default-settings
        fetch would pool Binance.com bars into a dataset measured to hold
        Binance.US ones, under fold windows and a cost scenario written for
        neither.
        """
        ds = DataService(data_root=tmp_path)
        provider = _NamedProvider("binance")

        with (
            patch.object(ds, "_get_provider", return_value=provider),
            pytest.raises(StorageVenueMismatchError, match="binance_us"),
        ):
            ds.fetch(
                "binance", "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), "1m", **self.SMOKE
            )

        assert provider.fetch_calls == 0
        assert not ds.store.exists(f"binance/{key_builder.bars('BTC_USDT', '1m')}")

    def test_refusal_precedes_the_provider_call(self, tmp_path: Path) -> None:
        """The venue is refused before the network is touched, not after."""
        ds = DataService(data_root=tmp_path)
        provider = _NamedProvider("coinbase")

        with (
            patch.object(ds, "_get_provider", return_value=provider),
            pytest.raises(StorageVenueMismatchError),
        ):
            ds.fetch("oanda", "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), "1m", **self.SMOKE)

        assert provider.fetch_calls == 0

    def test_backfill_is_guarded_too(self, tmp_path: Path) -> None:
        """``backfill`` reaches the provider directly on its gap-filling branch."""
        write_test_data(
            tmp_path,
            "binance",
            "BTC_USDT",
            "1m",
            pl.DataFrame(
                {
                    "timestamp": [
                        datetime(2024, 1, 15, 0, 0, tzinfo=UTC),
                        datetime(2024, 1, 15, 0, 5, tzinfo=UTC),
                    ],
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1.0, 1.0],
                }
            ),
        )
        ds = DataService(data_root=tmp_path)
        provider = _NamedProvider("binance")

        with (
            patch.object(ds, "_get_provider", return_value=provider),
            pytest.raises(StorageVenueMismatchError),
        ):
            ds.backfill(
                "binance",
                "BTC_USDT",
                date(2024, 1, 15),
                date(2024, 1, 16),
                "1m",
                **self.SMOKE,
            )

        assert provider.fetch_calls == 0

    def test_ordinary_providers_keep_key_equals_venue(self, tmp_path: Path) -> None:
        """No recorded exception means the plain rule, and it is the default."""
        ds = DataService(data_root=tmp_path)
        provider = _NamedProvider("oanda")

        with patch.object(ds, "_get_provider", return_value=provider):
            ds.fetch("oanda", "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), "1m", **self.SMOKE)

        assert provider.fetch_calls == 1

    def test_binance_us_cannot_be_populated_by_ingestion(self, tmp_path: Path) -> None:
        """The second key must not come into existence, so it is not ingestable.

        ``binance_us`` is now fold-governed, but a governed name is not a store
        key: no factory builds it, so ``fetch`` refuses with the clearer
        "Unknown provider" diagnostic that already existed. Together with the
        prohibition on copying or aliasing bars between keys, that is what keeps
        one venue in one key.
        """
        ds = DataService(data_root=tmp_path)
        assert "binance_us" not in DataService._PROVIDER_FACTORIES

        with pytest.raises(ValueError, match="Unknown provider"):
            ds.fetch(
                "binance_us",
                "BTC_USDT",
                date(2024, 1, 15),
                date(2024, 1, 15),
                "1m",
                **self.SMOKE,
            )
        assert not (tmp_path / "binance_us").exists()

    def test_the_real_binance_provider_reports_the_names_this_relies_on(self) -> None:
        """The venue check is only as good as the name the provider reports.

        One class, one boolean, two names. Asserted here rather than assumed,
        because every other assertion in this class depends on it.
        """
        assert BinanceProvider(use_us=True).name == "binance_us"
        assert BinanceProvider(use_us=False).name == "binance"

    def test_sync_is_guarded_too(self, tmp_path: Path) -> None:
        """``sync`` writes bars under a caller-chosen key like the others.

        Enforcing the invariant on ``fetch`` and ``backfill`` alone would leave
        the same hole in a different shape.
        """
        ds = DataService(data_root=tmp_path)
        provider = _NamedProvider("binance")

        with (
            patch.object(ds, "_get_provider", return_value=provider),
            pytest.raises(StorageVenueMismatchError),
        ):
            ds.sync(
                UniverseDefinition(
                    name="venue-check",
                    version=1,
                    kind=UniverseKind.EXPLICIT,
                    spec={"symbols": ["BTC_USDT"]},
                ),
                start=date(2024, 1, 15),
                end=date(2024, 1, 16),
                provider="binance",
                timeframe="1m",
                dataset="spot",
            )

        assert provider.fetch_calls == 0

    def test_a_double_without_a_string_name_is_not_checked(self, tmp_path: Path) -> None:
        """The documented tolerance, asserted rather than left to be discovered."""
        ds = DataService(data_root=tmp_path)
        provider = MagicMock()
        provider.fetch_bars.return_value = _one_bar()

        with patch.object(ds, "_get_provider", return_value=provider):
            ds.fetch(
                "binance", "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), "1m", **self.SMOKE
            )

        provider.fetch_bars.assert_called_once()

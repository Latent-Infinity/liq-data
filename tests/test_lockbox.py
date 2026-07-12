"""Tests for the research lockbox guard."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from liq.data.exceptions import LockboxViolationError, ValidationReuseError
from liq.data.lockbox import (
    FCE_LOCKBOX_LEDGER_V1,
    INTRADAY_CAMPAIGN_LEDGER_V1,
    LockboxGuard,
    resolve_dataset,
)
from liq.data.service import DataService
from liq.store import key_builder
from liq.store.parquet import ParquetStore


@pytest.fixture
def guard(tmp_path: Path) -> LockboxGuard:
    return LockboxGuard(usage_log_path=tmp_path / "lockbox_usage_log.jsonl")


def read_log(guard: LockboxGuard) -> list[dict]:
    text = guard.usage_log_path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestLedger:
    """The default ledger encodes the campaign fold boundaries."""

    def test_ledger_contains_all_campaign_datasets(self) -> None:
        expected = {
            "spy_qqq_ladder_tradestation",
            "tradestation_cohort_1m",
            "oanda_fx",
            "binance_spot",
            "binance_perp",
            "coinbase_spot",
            "databento_extended_hours",
        }
        assert set(INTRADAY_CAMPAIGN_LEDGER_V1.datasets) == expected

    def test_coinbase_spot_folds(self) -> None:
        windows = INTRADAY_CAMPAIGN_LEDGER_V1.datasets["coinbase_spot"]
        assert windows.discovery == (date(2020, 1, 1), date(2024, 12, 31))
        assert windows.validation == (date(2025, 1, 1), date(2025, 12, 31))
        assert windows.lockbox_start == date(2026, 1, 1)

    def test_ledger_has_version(self) -> None:
        assert INTRADAY_CAMPAIGN_LEDGER_V1.version


class TestFceLedger:
    """The FCE ledger freezes the approved 2020-2026 source envelope."""

    def test_fce_ledger_version_and_dataset_boundaries(self) -> None:
        assert FCE_LOCKBOX_LEDGER_V1.version == "fce_v1_2020_2026"
        expected = {
            "spy_qqq_ladder_tradestation",
            "tradestation_cohort_1m",
            "oanda_fx",
            "coinbase_spot",
            "databento_extended_hours",
            "fred_macro",
        }
        assert set(FCE_LOCKBOX_LEDGER_V1.datasets) == expected

        for windows in FCE_LOCKBOX_LEDGER_V1.datasets.values():
            assert windows.discovery == (date(2020, 1, 1), date(2024, 12, 31))
            assert windows.validation == (date(2025, 1, 1), date(2025, 12, 31))
            assert windows.lockbox_start == date(2026, 1, 1)

    def test_fce_ledger_blocks_ordinary_lockbox_reads(self, tmp_path: Path) -> None:
        guard = LockboxGuard(
            usage_log_path=tmp_path / "lockbox_usage_log.jsonl",
            ledger=FCE_LOCKBOX_LEDGER_V1,
        )

        with pytest.raises(LockboxViolationError, match="program lockbox"):
            guard.assert_period_allowed(
                "fred_macro",
                date(2026, 1, 1),
                date(2026, 12, 31),
                purpose="validation",
                arm_id="fce_macro_probe",
            )

    def test_fce_ledger_allows_human_only_final_review(self, tmp_path: Path) -> None:
        guard = LockboxGuard(
            usage_log_path=tmp_path / "lockbox_usage_log.jsonl",
            ledger=FCE_LOCKBOX_LEDGER_V1,
        )
        guard.assert_period_allowed(
            "oanda_fx",
            date(2026, 1, 1),
            date(2026, 3, 31),
            purpose="validation",
            arm_id="fce_portfolio_review",
            final_portfolio_review=True,
        )

        entry = read_log(guard)[0]
        assert entry["ledger_version"] == "fce_v1_2020_2026"
        assert entry["final_portfolio_review"] is True


class TestDiscovery:
    def test_discovery_read_within_window_allowed_and_logged(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "spy_qqq_ladder_tradestation",
            date(2015, 1, 1),
            date(2022, 12, 31),
            purpose="discovery",
            arm_id="idea_01",
        )
        entries = read_log(guard)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["dataset"] == "spy_qqq_ladder_tradestation"
        assert entry["purpose"] == "discovery"
        assert entry["arm_id"] == "idea_01"
        assert entry["start"] == "2015-01-01"
        assert entry["end"] == "2022-12-31"
        assert entry["final_portfolio_review"] is False
        assert entry["ledger_version"]

    def test_discovery_read_beyond_window_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2015, 1, 1),
                date(2025, 6, 30),
                purpose="discovery",
                arm_id="idea_01",
            )

    def test_discovery_on_dataset_without_discovery_folds_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "databento_extended_hours",
                date(2023, 6, 1),
                date(2023, 12, 31),
                purpose="discovery",
                arm_id="idea_02",
            )

    def test_violation_is_not_logged(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2015, 1, 1),
                date(2025, 6, 30),
                purpose="discovery",
                arm_id="idea_01",
            )
        assert not guard.usage_log_path.exists()


class TestLockboxPeriod:
    def test_lockbox_read_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2026, 1, 1),
                date(2026, 6, 30),
                purpose="discovery",
                arm_id="idea_01",
            )

    def test_window_overlapping_lockbox_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "oanda_fx",
                date(2020, 1, 1),
                date(2026, 3, 1),
                purpose="discovery",
                arm_id="idea_05a",
            )

    def test_final_portfolio_review_flag_allows_and_is_logged(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "spy_qqq_ladder_tradestation",
            date(2026, 1, 1),
            date(2026, 12, 31),
            purpose="validation",
            arm_id="portfolio_review",
            final_portfolio_review=True,
        )
        entries = read_log(guard)
        assert entries[0]["final_portfolio_review"] is True

    def test_final_portfolio_review_does_not_bypass_non_lockbox_windows(
        self, guard: LockboxGuard
    ) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2014, 1, 1),
                date(2014, 12, 31),
                purpose="discovery",
                arm_id="portfolio_review",
                final_portfolio_review=True,
            )

    def test_final_portfolio_review_requires_lockbox_only_window(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError, match="bounded"):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2025, 12, 1),
                date(2026, 1, 31),
                purpose="validation",
                arm_id="portfolio_review",
                final_portfolio_review=True,
            )

    def test_final_portfolio_review_keeps_dataset_purpose_restrictions(
        self, guard: LockboxGuard
    ) -> None:
        with pytest.raises(LockboxViolationError, match="only admits purposes"):
            guard.assert_period_allowed(
                "databento_extended_hours",
                date(2026, 1, 1),
                date(2026, 1, 31),
                purpose="validation",
                arm_id="portfolio_review",
                final_portfolio_review=True,
            )


class TestDevSmoke:
    def test_dev_smoke_allowed_anywhere_and_tagged(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "spy_qqq_ladder_tradestation",
            date(2025, 1, 1),
            date(2025, 12, 31),
            purpose="dev_smoke",
            arm_id="idea_01",
        )
        entries = read_log(guard)
        assert entries[0]["purpose"] == "dev_smoke"

    def test_dev_smoke_allowed_on_unassigned_dataset(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "binance_perp",
            date(2024, 1, 1),
            date(2024, 12, 31),
            purpose="dev_smoke",
            arm_id="idea_06b",
        )


class TestValidation:
    def test_validation_within_window_allowed(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "spy_qqq_ladder_tradestation",
            date(2025, 1, 1),
            date(2025, 12, 31),
            purpose="validation",
            arm_id="idea_01",
        )
        entries = read_log(guard)
        assert entries[0]["purpose"] == "validation"

    def test_validation_outside_window_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2022, 1, 1),
                date(2023, 12, 31),
                purpose="validation",
                arm_id="idea_01",
            )

    def test_second_validation_use_by_same_arm_raises(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        first = LockboxGuard(usage_log_path=log)
        first.assert_period_allowed(
            "spy_qqq_ladder_tradestation",
            date(2025, 1, 1),
            date(2025, 12, 31),
            purpose="validation",
            arm_id="idea_01",
        )
        second = LockboxGuard(usage_log_path=log)
        with pytest.raises(ValidationReuseError):
            second.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2025, 1, 1),
                date(2025, 12, 31),
                purpose="validation",
                arm_id="idea_01",
            )

    def test_repeated_validation_reads_within_same_session_allowed(
        self, guard: LockboxGuard
    ) -> None:
        for _ in range(3):
            guard.assert_period_allowed(
                "oanda_fx",
                date(2025, 1, 1),
                date(2025, 12, 31),
                purpose="validation",
                arm_id="idea_05a",
            )
        assert len(read_log(guard)) == 3

    def test_validation_use_by_different_arm_allowed(self, tmp_path: Path) -> None:
        log = tmp_path / "log.jsonl"
        first = LockboxGuard(usage_log_path=log)
        first.assert_period_allowed(
            "spy_qqq_ladder_tradestation",
            date(2025, 1, 1),
            date(2025, 12, 31),
            purpose="validation",
            arm_id="idea_01",
        )
        second = LockboxGuard(usage_log_path=log)
        second.assert_period_allowed(
            "spy_qqq_ladder_tradestation",
            date(2025, 1, 1),
            date(2025, 12, 31),
            purpose="validation",
            arm_id="idea_07",
        )


class TestDatabento:
    def test_characterization_within_window_allowed(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "databento_extended_hours",
            date(2023, 3, 1),
            date(2025, 12, 31),
            purpose="characterization",
            arm_id="idea_02",
        )

    def test_characterization_beyond_window_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "databento_extended_hours",
                date(2023, 3, 1),
                date(2026, 1, 31),
                purpose="characterization",
                arm_id="idea_02",
            )

    def test_forward_accrual_allowed_from_2026(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "databento_extended_hours",
            date(2026, 1, 1),
            date(2026, 6, 30),
            purpose="forward_accrual",
            arm_id="idea_02",
        )

    def test_forward_accrual_before_start_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "databento_extended_hours",
                date(2025, 6, 1),
                date(2026, 6, 30),
                purpose="forward_accrual",
                arm_id="idea_02",
            )

    def test_characterization_purpose_rejected_on_other_datasets(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2015, 1, 1),
                date(2022, 12, 31),
                purpose="characterization",
                arm_id="idea_01",
            )


class TestUnassignedAndUnknown:
    def test_research_read_on_unassigned_dataset_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "binance_perp",
                date(2021, 1, 1),
                date(2023, 12, 31),
                purpose="discovery",
                arm_id="idea_06b",
            )

    def test_unknown_dataset_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "no_such_dataset",
                date(2020, 1, 1),
                date(2020, 12, 31),
                purpose="discovery",
                arm_id="idea_01",
            )


class TestArgumentValidation:
    def test_unknown_purpose_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(ValueError, match="purpose"):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2015, 1, 1),
                date(2022, 12, 31),
                purpose="exploration",
                arm_id="idea_01",
            )

    def test_missing_bounds_raise(self, guard: LockboxGuard) -> None:
        with pytest.raises(ValueError, match="start and end"):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                None,
                date(2022, 12, 31),
                purpose="discovery",
                arm_id="idea_01",
            )

    def test_inverted_bounds_raise(self, guard: LockboxGuard) -> None:
        with pytest.raises(ValueError, match="start"):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2022, 12, 31),
                date(2015, 1, 1),
                purpose="discovery",
                arm_id="idea_01",
            )

    def test_empty_arm_id_raises(self, guard: LockboxGuard) -> None:
        with pytest.raises(ValueError, match="arm_id"):
            guard.assert_period_allowed(
                "spy_qqq_ladder_tradestation",
                date(2015, 1, 1),
                date(2022, 12, 31),
                purpose="discovery",
                arm_id="",
            )


class TestResolveDataset:
    @pytest.mark.parametrize(
        ("provider", "symbol", "expected"),
        [
            ("tradestation", "SPY", "spy_qqq_ladder_tradestation"),
            ("tradestation", "QQQ", "spy_qqq_ladder_tradestation"),
            ("tradestation", "AAPL", "tradestation_cohort_1m"),
            ("oanda", "EUR_USD", "oanda_fx"),
            ("binance", "BTC_USDT", "binance_spot"),
            ("coinbase", "BTC-USD", "coinbase_spot"),
            ("coinbase", "ETH-USD", "coinbase_spot"),
            ("databento", "AAPL", "databento_extended_hours"),
            ("fred", "DGS10", None),
            ("sec_edgar", "8K", None),
        ],
    )
    def test_mapping(self, provider: str, symbol: str, expected: str | None) -> None:
        assert resolve_dataset(provider, symbol) == expected


class TestDataServiceIntegration:
    """DataService.load routes research reads through the guard."""

    @staticmethod
    def _write_bars(tmp_path: Path, provider: str, symbol: str, ts: datetime) -> None:
        store = ParquetStore(str(tmp_path))
        storage_key = f"{provider}/{key_builder.bars(symbol, '1m')}"
        df = pl.DataFrame(
            {
                "timestamp": [ts],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [100.0],
            }
        )
        store.write(storage_key, df)

    def test_load_with_purpose_passes_guard_and_logs(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2020, 6, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)
        df = ds.load(
            "oanda",
            "EUR_USD",
            "1m",
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            purpose="discovery",
            arm_id="idea_05a",
        )
        assert isinstance(df, pl.DataFrame)
        log_path = tmp_path / "lockbox_usage_log.jsonl"
        assert log_path.exists()
        entry = json.loads(log_path.read_text().splitlines()[0])
        assert entry["dataset"] == "oanda_fx"
        assert entry["purpose"] == "discovery"

    def test_load_lockbox_period_raises(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2026, 2, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)
        with pytest.raises(LockboxViolationError):
            ds.load(
                "oanda",
                "EUR_USD",
                "1m",
                start=date(2026, 1, 1),
                end=date(2026, 12, 31),
                purpose="discovery",
                arm_id="idea_05a",
            )

    def test_load_research_read_requires_bounds(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2020, 6, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)
        with pytest.raises(ValueError, match="start and end"):
            ds.load("oanda", "EUR_USD", "1m", purpose="discovery", arm_id="idea_05a")

    def test_load_purpose_requires_arm_id(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2020, 6, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)
        with pytest.raises(ValueError, match="arm_id"):
            ds.load(
                "oanda",
                "EUR_USD",
                "1m",
                start=date(2020, 1, 1),
                end=date(2020, 12, 31),
                purpose="discovery",
            )

    def test_load_without_purpose_does_not_log(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2020, 6, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)
        ds.load("oanda", "EUR_USD", "1m")
        assert not (tmp_path / "lockbox_usage_log.jsonl").exists()

    def test_load_non_campaign_provider_with_purpose_is_not_guarded(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "alpaca", "AAPL", datetime(2026, 2, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)
        df = ds.load(
            "alpaca",
            "AAPL",
            "1m",
            start=date(2026, 1, 1),
            end=date(2026, 12, 31),
            purpose="discovery",
            arm_id="idea_99",
        )
        assert isinstance(df, pl.DataFrame)

    def test_load_can_use_fce_ledger(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2025, 6, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path, lockbox_ledger=FCE_LOCKBOX_LEDGER_V1)
        ds.load(
            "oanda",
            "EUR_USD",
            "1m",
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            purpose="validation",
            arm_id="fce_timestamp_contract",
        )

        entry = json.loads((tmp_path / "lockbox_usage_log.jsonl").read_text().splitlines()[0])
        assert entry["ledger_version"] == "fce_v1_2020_2026"

    def test_fce_ledger_load_blocks_ordinary_2026_reads(self, tmp_path: Path) -> None:
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2026, 2, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path, lockbox_ledger=FCE_LOCKBOX_LEDGER_V1)

        with pytest.raises(LockboxViolationError, match="program lockbox"):
            ds.load(
                "oanda",
                "EUR_USD",
                "1m",
                start=date(2026, 1, 1),
                end=date(2026, 12, 31),
                purpose="validation",
                arm_id="fce_timestamp_contract",
            )

"""Tests for the research lockbox guard."""

import ast
import inspect
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

import liq.data.lockbox as lockbox_module
from liq.data.exceptions import (
    LockboxViolationError,
    UnclassifiedProviderError,
    UndeclaredReadError,
    ValidationReuseError,
)
from liq.data.lockbox import (
    FCE_LOCKBOX_LEDGER_V1,
    INTRADAY_CAMPAIGN_LEDGER_V1,
    RESOLVABLE_DATASETS,
    STORAGE_KEY_VENUES,
    UNGOVERNED_PROVIDERS,
    LockboxGuard,
    LockboxLedger,
    declared_storage_venue,
    resolve_dataset,
    ungoverned_dataset_identity,
    unreachable_datasets,
)
from liq.data.service import DataService
from liq.store import key_builder
from liq.store.parquet import ParquetStore

#: The exemplar this suite uses wherever it needs a provider nobody classified.
#:
#: It was ``binance_us`` until
#: ``liq-docs/plans/oracle-binance-us-market-identity-2026-09-24.md`` mapped that
#: string to ``binance_spot``. A mapped provider asserting "nobody has
#: classified this" would be a test claiming something false, so the role moved
#: to a string that genuinely holds it rather than the tests being deleted.
#: ``test_the_exemplar_is_genuinely_unclassified`` fails the moment somebody
#: classifies this one, which is the signal to move the role again.
UNCLASSIFIED_PROVIDER = "kraken"


@pytest.fixture
def guard(tmp_path: Path) -> LockboxGuard:
    return LockboxGuard(usage_log_path=tmp_path / "lockbox_usage_log.jsonl")


def read_log(guard: LockboxGuard) -> list[dict]:
    text = guard.usage_log_path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestHermeticUsageLedger:
    """Running the suite must not append to the real usage ledger."""

    def test_default_data_root_is_a_throwaway_directory(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """A default-constructed service keeps its ledger under pytest's tmp base.

        ``DATA_ROOT`` in the developer's ``.env`` names the production store,
        so without the autouse override in ``conftest`` every declared read the
        suite makes through a default-constructed service appends to the
        append-only ledger that lives there. Those entries cannot be removed
        afterwards, which is why this is asserted rather than assumed.
        """
        service = DataService()

        assert service.data_root.is_relative_to(tmp_path_factory.getbasetemp())
        assert service.lockbox_guard.usage_log_path.is_relative_to(tmp_path_factory.getbasetemp())


class TestLedger:
    """The default ledger encodes the campaign fold boundaries."""

    def test_ledger_contains_all_campaign_datasets(self) -> None:
        expected = {
            "spy_qqq_ladder_tradestation",
            "month_end_spy_agg",
            "index_recon_sp500",
            "short_interest_bar_coverage",
            "tradestation_cohort_1m",
            "oanda_fx",
            "binance_spot",
            "binance_perp",
            "coinbase_spot",
            "databento_extended_hours",
            "tradestation_crypto_futures",
        }
        assert set(INTRADAY_CAMPAIGN_LEDGER_V1.datasets) == expected

    def test_coinbase_spot_folds(self) -> None:
        windows = INTRADAY_CAMPAIGN_LEDGER_V1.datasets["coinbase_spot"]
        assert windows.discovery == (date(2020, 1, 1), date(2024, 12, 31))
        assert windows.validation == (date(2025, 1, 1), date(2025, 12, 31))
        assert windows.lockbox_start == date(2026, 1, 1)

    def test_ledger_has_version(self) -> None:
        assert INTRADAY_CAMPAIGN_LEDGER_V1.version


class TestMonthEndDataset:
    def test_month_end_book_routes_spy_and_agg(self) -> None:
        assert (
            resolve_dataset("tradestation", "SPY", asset_class="month_end_book")
            == "month_end_spy_agg"
        )
        assert (
            resolve_dataset("tradestation", "AGG", asset_class="month_end_book")
            == "month_end_spy_agg"
        )

    def test_agg_without_hint_falls_to_cohort(self) -> None:
        assert resolve_dataset("tradestation", "AGG") == "tradestation_cohort_1m"

    def test_month_end_discovery_2017_allowed(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "month_end_spy_agg",
            date(2017, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="month_end_rebalance",
        )

    def test_cohort_rejects_2017_discovery(self, guard: LockboxGuard) -> None:
        # Without the dedicated dataset, AGG's cohort window (2019+) rejects 2017.
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "tradestation_cohort_1m",
                date(2017, 1, 1),
                date(2018, 12, 31),
                purpose="discovery",
                arm_id="month_end_rebalance",
            )


class TestIndexReconDataset:
    def test_index_recon_book_routes_any_symbol(self) -> None:
        assert (
            resolve_dataset("tradestation", "SPY", asset_class="index_recon_book")
            == "index_recon_sp500"
        )
        assert (
            resolve_dataset("tradestation", "AAPL", asset_class="index_recon_book")
            == "index_recon_sp500"
        )
        assert (
            resolve_dataset("alpaca", "AAPL", asset_class="index_recon_book") == "index_recon_sp500"
        )
        assert (
            resolve_dataset("sec_edgar", "AAPL", asset_class="index_recon_book")
            == "index_recon_sp500"
        )

    def test_alpaca_without_index_recon_hint_is_not_campaign_data(self) -> None:
        assert resolve_dataset("alpaca", "AAPL") is None

    def test_symbol_without_hint_falls_to_cohort(self) -> None:
        assert resolve_dataset("tradestation", "AAPL") == "tradestation_cohort_1m"

    def test_index_recon_folds(self) -> None:
        windows = INTRADAY_CAMPAIGN_LEDGER_V1.datasets["index_recon_sp500"]
        assert windows.discovery == (date(2017, 1, 1), date(2024, 12, 31))
        assert windows.validation == (date(2025, 1, 1), date(2025, 12, 31))
        assert windows.lockbox_start == date(2026, 1, 1)

    def test_index_recon_discovery_2017_allowed(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "index_recon_sp500",
            date(2017, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="index_reconstitution",
        )

    def test_index_recon_discovery_before_2017_rejected(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "index_recon_sp500",
                date(2016, 1, 1),
                date(2016, 12, 31),
                purpose="discovery",
                arm_id="index_reconstitution",
            )

    def test_index_recon_lockbox_read_rejected(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError, match="program lockbox"):
            guard.assert_period_allowed(
                "index_recon_sp500",
                date(2025, 1, 1),
                date(2026, 1, 1),
                purpose="validation",
                arm_id="index_reconstitution",
            )


class TestShortInterestBarCoverageDataset:
    def test_short_interest_book_routes_to_dedicated_dataset(self) -> None:
        assert (
            resolve_dataset("tradestation", "OLD", asset_class="short_interest_book")
            == "short_interest_bar_coverage"
        )

    def test_short_interest_book_has_only_authorized_characterization_window(self) -> None:
        windows = INTRADAY_CAMPAIGN_LEDGER_V1.datasets["short_interest_bar_coverage"]

        assert windows.characterization == (date(2021, 6, 15), date(2025, 12, 31))
        assert windows.allowed_purposes == frozenset({"characterization"})
        assert windows.discovery is None
        assert windows.validation is None
        assert windows.lockbox_start is None

    def test_short_interest_characterization_window_is_admitted(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "short_interest_bar_coverage",
            date(2021, 6, 15),
            date(2025, 12, 31),
            purpose="characterization",
            arm_id="path_d_short_interest",
        )

    def test_short_interest_dataset_rejects_discovery(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError, match="only admits purposes"):
            guard.assert_period_allowed(
                "short_interest_bar_coverage",
                date(2021, 6, 15),
                date(2024, 12, 31),
                purpose="discovery",
                arm_id="path_d_short_interest",
            )

    def test_short_interest_dataset_rejects_out_of_window_read(self, guard: LockboxGuard) -> None:
        with pytest.raises(LockboxViolationError, match="outside the characterization window"):
            guard.assert_period_allowed(
                "short_interest_bar_coverage",
                date(2021, 6, 14),
                date(2025, 12, 31),
                purpose="characterization",
                arm_id="path_d_short_interest",
            )


class TestDatabentoIndexReconRouting:
    """Databento daily-bar reads declared as index_recon_book share the
    index-reconstitution dataset and its fold governance, exactly as the
    TradeStation routing does."""

    def test_index_recon_book_routes_any_symbol(self) -> None:
        assert (
            resolve_dataset("databento", "SPY", asset_class="index_recon_book")
            == "index_recon_sp500"
        )
        assert (
            resolve_dataset("databento", "AAPL", asset_class="index_recon_book")
            == "index_recon_sp500"
        )

    def test_symbol_without_hint_keeps_extended_hours(self) -> None:
        assert resolve_dataset("databento", "AAPL") == "databento_extended_hours"

    def test_option_hint_still_routes_to_opra(self) -> None:
        assert resolve_dataset("databento", "SPY", asset_class="option") == "databento_opra_options"

    def test_extended_hours_refuses_discovery(self, guard: LockboxGuard) -> None:
        # The default databento dataset admits no discovery reads; the
        # index_recon_book hint is what makes guarded discovery possible.
        with pytest.raises(LockboxViolationError, match="only admits purposes"):
            guard.assert_period_allowed(
                "databento_extended_hours",
                date(2023, 1, 1),
                date(2024, 12, 31),
                purpose="discovery",
                arm_id="index_reconstitution",
            )

    def test_index_recon_discovery_2017_allowed(self, guard: LockboxGuard) -> None:
        dataset = resolve_dataset("databento", "AAPL", asset_class="index_recon_book")
        assert dataset is not None
        guard.assert_period_allowed(
            dataset,
            date(2017, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="index_reconstitution",
        )

    def test_index_recon_discovery_before_2017_rejected(self, guard: LockboxGuard) -> None:
        dataset = resolve_dataset("databento", "AAPL", asset_class="index_recon_book")
        assert dataset is not None
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                dataset,
                date(2016, 1, 1),
                date(2016, 12, 31),
                purpose="discovery",
                arm_id="index_reconstitution",
            )

    def test_index_recon_validation_period_refused_for_discovery(self, guard: LockboxGuard) -> None:
        dataset = resolve_dataset("databento", "AAPL", asset_class="index_recon_book")
        assert dataset is not None
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                dataset,
                date(2025, 1, 1),
                date(2025, 12, 31),
                purpose="discovery",
                arm_id="index_reconstitution",
            )

    def test_index_recon_lockbox_read_rejected(self, guard: LockboxGuard) -> None:
        dataset = resolve_dataset("databento", "AAPL", asset_class="index_recon_book")
        assert dataset is not None
        with pytest.raises(LockboxViolationError, match="program lockbox"):
            guard.assert_period_allowed(
                dataset,
                date(2025, 1, 1),
                date(2026, 1, 1),
                purpose="validation",
                arm_id="index_reconstitution",
            )

    def test_tradestation_routing_unchanged(self) -> None:
        assert (
            resolve_dataset("tradestation", "AAPL", asset_class="index_recon_book")
            == "index_recon_sp500"
        )
        assert resolve_dataset("tradestation", "AAPL") == "tradestation_cohort_1m"

    @pytest.mark.parametrize("provider", ["tradestation", "databento"])
    def test_research_service_rejects_non_daily_index_recon_reads(
        self, tmp_path: Path, provider: str
    ) -> None:
        service = DataService(data_root=tmp_path)
        with pytest.raises(LockboxViolationError, match="daily bars only"):
            service.load(
                provider,
                "AAPL",
                "1m",
                start=date(2020, 1, 1),
                end=date(2020, 12, 31),
                purpose="discovery",
                arm_id="index_reconstitution",
                asset_class="index_recon_book",
            )


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
            "databento_opra_options",
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


class TestUnboundedDevSmoke:
    """An unbounded read is admissible for ``dev_smoke`` only, and is recorded
    as unbounded rather than given a window it did not have.

    ``dev_smoke`` is fold-exempt by design, so requiring bounds on it never
    prevented lockbox exposure; what bounds buy is a reconcilable ledger entry.
    An entry that says "unbounded" is honest; a fabricated range is not.
    """

    @pytest.mark.parametrize(
        ("start", "end"),
        [(None, None), (date(2024, 1, 1), None), (None, date(2024, 12, 31))],
    )
    def test_unbounded_dev_smoke_is_logged_as_unbounded(
        self, guard: LockboxGuard, start: date | None, end: date | None
    ) -> None:
        guard.assert_period_allowed(
            "tradestation_cohort_1m",
            start,
            end,
            purpose="dev_smoke",
            arm_id="liq_data_service_diagnostics",
        )
        entry = read_log(guard)[0]
        assert entry["unbounded"] is True
        assert entry["start"] == (start.isoformat() if start else None)
        assert entry["end"] == (end.isoformat() if end else None)

    def test_bounded_entries_carry_no_unbounded_marker(self, guard: LockboxGuard) -> None:
        """A bounded entry keeps exactly the shape every historical entry has."""
        guard.assert_period_allowed(
            "tradestation_cohort_1m",
            date(2024, 1, 1),
            date(2024, 12, 31),
            purpose="dev_smoke",
            arm_id="liq_data_service_diagnostics",
        )
        assert "unbounded" not in read_log(guard)[0]

    @pytest.mark.parametrize(
        "purpose", ["discovery", "validation", "characterization", "forward_accrual"]
    )
    def test_research_purposes_still_require_bounds(
        self, guard: LockboxGuard, purpose: str
    ) -> None:
        with pytest.raises(ValueError, match="start and end"):
            guard.assert_period_allowed(
                "tradestation_cohort_1m",
                None,
                None,
                purpose=purpose,
                arm_id="idea_01",
            )
        assert not guard.usage_log_path.exists()

    def test_unbounded_dev_smoke_is_not_citable_coverage(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "tradestation_cohort_1m",
            None,
            None,
            purpose="dev_smoke",
            arm_id="idea_01",
        )
        assert guard.guarded_windows(arm_id="idea_01", dataset="tradestation_cohort_1m") == []


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
            ("binance_us", "BTC_USDT", "binance_spot"),
            ("coinbase", "BTC-USD", "coinbase_spot"),
            ("coinbase", "ETH-USD", "coinbase_spot"),
            ("databento", "AAPL", "databento_extended_hours"),
            ("fred", "DGS10", None),
            ("sec_edgar", "8K", None),
        ],
    )
    def test_mapping(self, provider: str, symbol: str, expected: str | None) -> None:
        assert resolve_dataset(provider, symbol) == expected

    def test_databento_option_asset_class_routes_to_opra(self) -> None:
        # Options reads must resolve to their own fold-governed dataset so the
        # guard can distinguish an OPRA read from an equity-bars read.
        assert resolve_dataset("databento", "SPY", asset_class="option") == "databento_opra_options"

    @pytest.mark.parametrize("asset_class", [None, "equity", "future"])
    def test_databento_non_option_asset_class_stays_bars(self, asset_class: str | None) -> None:
        assert (
            resolve_dataset("databento", "AAPL", asset_class=asset_class)
            == "databento_extended_hours"
        )

    def test_asset_class_ignored_for_providers_without_a_discriminator(self) -> None:
        assert resolve_dataset("oanda", "EUR_USD", asset_class="option") == "oanda_fx"

    def test_tradestation_future_asset_class_routes_to_crypto_futures(self) -> None:
        # Mirrors the databento option discriminator: a futures read must be
        # fold-governed independently of the equity cohort.
        assert (
            resolve_dataset("tradestation", "@MET", asset_class="future")
            == "tradestation_crypto_futures"
        )

    @pytest.mark.parametrize("symbol", ["@MBT", "@MET"])
    def test_unambiguous_continuous_crypto_symbols_route_without_hint(self, symbol: str) -> None:
        """Known continuous crypto roots must not silently borrow equity folds."""
        assert resolve_dataset("tradestation", symbol) == "tradestation_crypto_futures"

    @pytest.mark.parametrize("symbol", ["@MET", "@MBT", "MET", "MBT"])
    def test_futures_read_never_resolves_to_the_equity_cohort(self, symbol: str) -> None:
        """The bug this closes: MET is micro ether at the CME and MetLife in equities.

        Without the discriminator a crypto-futures read resolved to
        ``tradestation_cohort_1m`` and was silently governed by the equity
        cohort's fold windows.
        """
        resolved = resolve_dataset("tradestation", symbol, asset_class="future")
        assert resolved != "tradestation_cohort_1m"
        assert resolved == "tradestation_crypto_futures"

    def test_future_asset_class_wins_over_the_ladder_symbols(self) -> None:
        """A declared futures read is a futures read whatever the ticker looks like."""
        assert (
            resolve_dataset("tradestation", "SPY", asset_class="future")
            == "tradestation_crypto_futures"
        )

    @pytest.mark.parametrize("asset_class", [None, "equity", "option"])
    def test_tradestation_non_future_asset_class_is_unchanged(
        self, asset_class: str | None
    ) -> None:
        """Backward compatibility: every existing caller keeps its mapping."""
        assert (
            resolve_dataset("tradestation", "AAPL", asset_class=asset_class)
            == "tradestation_cohort_1m"
        )
        assert (
            resolve_dataset("tradestation", "SPY", asset_class=asset_class)
            == "spy_qqq_ladder_tradestation"
        )


class TestBinanceUsIsTheSameSurfaceAsBinanceSpot:
    """One venue under two strings, mapped to the one dataset that governs it.

    ``liq-docs/plans/oracle-binance-us-market-identity-2026-09-24.md`` measured
    the on-disk ``binance`` store against a live ``api.binance.us`` pull:
    11,520 minutes over two symbols and four years, 100.00% identical on close
    and volume, with ``api.binance.com`` returning HTTP 451 from this host. The
    ``binance_spot`` dataset already holds Binance.US bars, so mapping
    ``binance_us`` to it records what the data is. Refusing to map was the
    claim, and it asserted the two strings name different markets.
    """

    def test_both_strings_resolve_to_the_one_dataset(self) -> None:
        assert resolve_dataset("binance_us", "BTC_USDT") == "binance_spot"
        assert resolve_dataset("binance_us", "BTC_USDT") == resolve_dataset("binance", "BTC_USDT")

    @pytest.mark.parametrize("spelling", ["BINANCE_US", "Binance_US", "binance_US"])
    def test_case_is_pinned_not_assumed(self, spelling: str) -> None:
        """The resolver lowercases its input; that is asserted, not trusted."""
        assert resolve_dataset(spelling, "BTC_USDT") == "binance_spot"

    def test_mapping_did_not_touch_the_ungoverned_allow_list(self) -> None:
        """A mapped provider is not an ungoverned one, and the list is untouched."""
        assert "binance_us" not in UNGOVERNED_PROVIDERS
        assert set(UNGOVERNED_PROVIDERS) == {"fred", "polygon"}

    def test_no_second_dataset_was_created_for_the_venue(self) -> None:
        """Two datasets over the same bars would be two fold regimes over them.

        The prohibition is explicit: no ``binance_us`` dataset, no fold windows
        of its own, no cost scenario of its own.
        """
        for ledger in (INTRADAY_CAMPAIGN_LEDGER_V1, FCE_LOCKBOX_LEDGER_V1):
            assert "binance_us" not in ledger.datasets
        assert "binance_us" not in RESOLVABLE_DATASETS


class TestStorageKeyVenues:
    """A storage key is a claim about the venue its bars came from."""

    def test_default_is_the_key_itself(self) -> None:
        """Every provider without a recorded exception claims its own name."""
        for provider in ("oanda", "tradestation", "coinbase", "databento", "fred"):
            assert declared_storage_venue(provider) == provider

    def test_the_binance_key_is_declared_to_hold_binance_us_bars(self) -> None:
        assert declared_storage_venue("binance") == "binance_us"
        assert declared_storage_venue("BINANCE") == "binance_us"

    def test_every_entry_carries_a_reason(self) -> None:
        """As with the ungoverned allow-list, membership is a recorded decision."""
        for key, record in STORAGE_KEY_VENUES.items():
            assert key == key.lower()
            assert record.provider_name == record.provider_name.lower()
            assert len(record.reason.split()) >= 10

    def test_registry_cannot_be_mutated_at_runtime(self) -> None:
        with pytest.raises(TypeError):
            STORAGE_KEY_VENUES["kraken"] = None  # type: ignore[index]


class TestUngovernedProviderAllowList:
    """ "Ungoverned" is a recorded decision, not the absence of a branch."""

    def test_membership_is_exactly_the_recorded_allow_list(self) -> None:
        assert set(UNGOVERNED_PROVIDERS) == {"fred", "polygon"}

    @pytest.mark.parametrize("provider", sorted(UNGOVERNED_PROVIDERS))
    def test_every_entry_carries_a_reason(self, provider: str) -> None:
        """A bare membership list would restore the thing it replaces.

        The list exists so that "ungoverned" names a decision somebody took.
        An entry with no stated reason is indistinguishable from a provider
        nobody classified, which is the state the allow-list ends.
        """
        reason = UNGOVERNED_PROVIDERS[provider]
        assert reason.strip()
        assert len(reason.split()) >= 10

    def test_allow_list_cannot_be_mutated_at_runtime(self) -> None:
        with pytest.raises(TypeError):
            UNGOVERNED_PROVIDERS[UNCLASSIFIED_PROVIDER] = "convenient"  # type: ignore[index]

    def test_identity_is_stable_lowercased_and_never_a_ledger_dataset(self) -> None:
        assert ungoverned_dataset_identity("FRED") == "ungoverned:fred"
        assert ungoverned_dataset_identity("fred") == "ungoverned:fred"
        for ledger in (INTRADAY_CAMPAIGN_LEDGER_V1, FCE_LOCKBOX_LEDGER_V1):
            assert ungoverned_dataset_identity("fred") not in ledger.datasets


def _dataset_names_the_resolver_can_return() -> frozenset[str]:
    """Every string literal ``resolve_dataset`` returns, read from its source.

    Derived rather than restated so ``RESOLVABLE_DATASETS`` cannot drift away
    from the function it describes when a mapping branch is added.
    """
    tree = ast.parse(inspect.getsource(lockbox_module))
    resolver = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_dataset"
    )
    return frozenset(
        node.value.value
        for node in ast.walk(resolver)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


class TestLedgerReachability:
    """A ledger that declares a dataset no read can route to governs nothing."""

    def test_resolvable_datasets_matches_the_resolver_source(self) -> None:
        assert _dataset_names_the_resolver_can_return() == RESOLVABLE_DATASETS

    @pytest.mark.parametrize(
        "ledger",
        [
            pytest.param(
                INTRADAY_CAMPAIGN_LEDGER_V1,
                id="intraday_campaign",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "'binance_perp' is declared with empty fold windows pending perp "
                        "depth verification and no resolve_dataset call can return it, so "
                        "no read can reach the governance the ledger wrote down. Routing it "
                        "is an open decision for the campaign ledger's owner "
                        "(oracle-unmapped-provider-bypass-2026-09-23.md sec.7); mapping it "
                        "to make this test pass is prohibited by that record."
                    ),
                ),
            ),
            pytest.param(
                FCE_LOCKBOX_LEDGER_V1,
                id="fce",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "'fred_macro' is declared with instrument-bar fold windows it "
                        "inherited structurally and no resolve_dataset call can return it. "
                        "Wiring it on those windows would refuse the existing legitimate "
                        "2014-2023 macro reads; designing macro-appropriate windows is an "
                        "open decision for the FCE ledger's owner "
                        "(oracle-unmapped-provider-bypass-2026-09-23.md sec.7)."
                    ),
                ),
            ),
        ],
    )
    def test_every_declared_dataset_is_reachable_by_the_resolver(
        self, ledger: LockboxLedger
    ) -> None:
        """Both cases fail today, and the failure is the point.

        Two datasets carry governance nobody can reach. The test is the standing
        record that the routing decisions are owed; it is marked strict so that
        taking either decision turns it green loudly rather than silently.
        """
        assert unreachable_datasets(ledger) == frozenset()

    def test_unreachable_datasets_names_exactly_the_owed_decisions(self) -> None:
        assert unreachable_datasets(INTRADAY_CAMPAIGN_LEDGER_V1) == frozenset({"binance_perp"})
        assert unreachable_datasets(FCE_LOCKBOX_LEDGER_V1) == frozenset({"fred_macro"})


class TestResolverLedgerReconciliation:
    """The resolver is ledger-blind; the guard must not hold that silently."""

    def test_classify_returns_a_dataset_the_active_ledger_declares(
        self, guard: LockboxGuard
    ) -> None:
        assert guard.classify_read("oanda", "EUR_USD") == "oanda_fx"

    def test_classify_returns_none_when_no_dataset_governs_the_read(
        self, guard: LockboxGuard
    ) -> None:
        assert guard.classify_read("fred", "TB3MS") is None

    def test_classify_fails_loudly_when_the_ledger_lacks_the_resolved_dataset(
        self, tmp_path: Path
    ) -> None:
        """The FCE ledger declares no index-reconstitution dataset.

        Before this check the disagreement travelled as far as
        ``assert_period_allowed`` and, on the ungoverned path, not at all. A
        resolver answer the active ledger does not recognise is a configuration
        fault, and it is named as one.
        """
        guard = LockboxGuard(
            usage_log_path=tmp_path / "lockbox_usage_log.jsonl",
            ledger=FCE_LOCKBOX_LEDGER_V1,
        )
        with pytest.raises(LockboxViolationError, match="fce_v1_2020_2026"):
            guard.classify_read("alpaca", "AAPL", asset_class="index_recon_book")


class TestUngovernedReadLedgerEntry:
    """A declared read that no dataset governs is still recorded."""

    def test_entry_records_null_dataset_governed_false_and_the_provider(
        self, guard: LockboxGuard
    ) -> None:
        guard.record_ungoverned_read(
            "fred",
            date(2014, 1, 1),
            date(2023, 1, 1),
            purpose="discovery",
            arm_id="idea_09_conditioned_overlay",
        )

        entry = read_log(guard)[0]
        assert entry["dataset"] is None
        assert entry["governed"] is False
        assert entry["provider"] == "fred"
        assert entry["ungoverned_dataset"] == "ungoverned:fred"
        assert entry["purpose"] == "discovery"
        assert entry["arm_id"] == "idea_09_conditioned_overlay"
        assert entry["start"] == "2014-01-01"
        assert entry["end"] == "2023-01-01"
        assert entry["ledger_version"] == INTRADAY_CAMPAIGN_LEDGER_V1.version

    def test_entry_can_never_cover_a_governed_dataset_claim(self, guard: LockboxGuard) -> None:
        """The laundering check, verified rather than assumed.

        ``guarded_windows`` filters on ``entry["dataset"] != dataset``. A null
        dataset compares unequal to every dataset name, so no ungoverned entry
        can supply citable coverage for any governed dataset -- including one
        whose windows would have contained the ungoverned read's period.
        """
        guard.record_ungoverned_read(
            "fred",
            date(2020, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="arm_x",
        )

        for dataset in INTRADAY_CAMPAIGN_LEDGER_V1.datasets:
            assert guard.guarded_windows(arm_id="arm_x", dataset=dataset) == []
        assert guard.guarded_windows(arm_id="arm_x", dataset="ungoverned:fred") == []
        assert guard.guarded_windows(arm_id="arm_x", dataset="fred_macro") == []

    def test_entry_is_attributable_under_its_synthetic_identity(self, guard: LockboxGuard) -> None:
        """Attribution is the other half: silence was the previous only option.

        A runner that read FRED could neither name the read (nothing to
        attribute it to) nor omit it (provenance would be incomplete), so it
        omitted it. The synthetic identity gives the honest third answer.
        """
        guard.record_ungoverned_read(
            "FRED",
            date(2014, 1, 1),
            date(2023, 1, 1),
            purpose="discovery",
            arm_id="idea_11_tax_loss_harvesting",
        )

        assert guard.ungoverned_windows(arm_id="idea_11_tax_loss_harvesting", provider="fred") == [
            ("2014-01-01", "2023-01-01")
        ]
        assert guard.ungoverned_windows(arm_id="somebody_else", provider="fred") == []
        assert (
            guard.ungoverned_windows(arm_id="idea_11_tax_loss_harvesting", provider="polygon") == []
        )

    def test_attribution_excludes_dev_smoke_exactly_as_citable_coverage_does(
        self, guard: LockboxGuard
    ) -> None:
        guard.record_ungoverned_read(
            "fred",
            date(2014, 1, 1),
            date(2023, 1, 1),
            purpose="dev_smoke",
            arm_id="arm_x",
        )
        assert guard.ungoverned_windows(arm_id="arm_x", provider="fred") == []
        assert len(read_log(guard)) == 1

    def test_attribution_can_be_narrowed_to_the_current_session(self, tmp_path: Path) -> None:
        log_path = tmp_path / "lockbox_usage_log.jsonl"
        first = LockboxGuard(usage_log_path=log_path)
        second = LockboxGuard(usage_log_path=log_path)
        for guard in (first, second):
            guard.record_ungoverned_read(
                "fred",
                date(2014, 1, 1),
                date(2023, 1, 1),
                purpose="discovery",
                arm_id="arm_x",
            )

        assert len(second.ungoverned_windows(arm_id="arm_x", provider="fred")) == 2
        assert second.ungoverned_windows(
            arm_id="arm_x", provider="fred", current_session_only=True
        ) == [("2014-01-01", "2023-01-01")]

    def test_research_purpose_still_requires_bounds(self, guard: LockboxGuard) -> None:
        """An unbounded read is the read the guard exists to make impossible.

        Ungoverned reads bypassed the bounds requirement along with everything
        else below the early return. Nothing about being ungoverned makes an
        unbounded research read checkable or attributable.
        """
        with pytest.raises(ValueError, match="start and end"):
            guard.record_ungoverned_read("fred", None, None, purpose="discovery", arm_id="arm_x")
        assert not guard.usage_log_path.exists()

    def test_unbounded_dev_smoke_read_is_logged_as_unbounded(self, guard: LockboxGuard) -> None:
        guard.record_ungoverned_read("fred", None, None, purpose="dev_smoke", arm_id="arm_x")

        entry = read_log(guard)[0]
        assert entry["unbounded"] is True
        assert entry["start"] is None
        assert entry["end"] is None
        assert entry["governed"] is False

    def test_unknown_purpose_is_refused(self, guard: LockboxGuard) -> None:
        with pytest.raises(ValueError, match="unknown purpose"):
            guard.record_ungoverned_read(
                "fred",
                date(2014, 1, 1),
                date(2023, 1, 1),
                purpose="not_a_real_purpose",
                arm_id="arm_x",
            )

    def test_empty_arm_id_is_refused(self, guard: LockboxGuard) -> None:
        with pytest.raises(ValueError, match="arm_id"):
            guard.record_ungoverned_read(
                "fred", date(2014, 1, 1), date(2023, 1, 1), purpose="discovery", arm_id=""
            )

    def test_inverted_bounds_are_refused(self, guard: LockboxGuard) -> None:
        with pytest.raises(ValueError, match="start"):
            guard.record_ungoverned_read(
                "fred", date(2023, 1, 1), date(2014, 1, 1), purpose="discovery", arm_id="arm_x"
            )


class TestTradeStationCryptoFutures:
    """Crypto futures fold windows frozen by the basis-carry registration."""

    def test_dataset_is_registered_with_frozen_windows(self) -> None:
        windows = INTRADAY_CAMPAIGN_LEDGER_V1.datasets["tradestation_crypto_futures"]
        assert windows.discovery == (date(2022, 1, 1), date(2024, 12, 31))
        assert windows.validation == (date(2025, 1, 1), date(2025, 12, 31))
        assert windows.lockbox_start == date(2026, 1, 1)

    def test_discovery_read_in_window_allowed(self, tmp_path: Path) -> None:
        guard = LockboxGuard(tmp_path / "usage.jsonl")
        guard.assert_period_allowed(
            "tradestation_crypto_futures",
            date(2022, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="path_e_crypto_basis_carry",
        )
        assert (tmp_path / "usage.jsonl").exists()

    def test_pre_inception_discovery_read_rejected(self, tmp_path: Path) -> None:
        """2021 precedes the frozen 2022 discovery start (ragged @MET); the
        tail is excluded, never backfilled (PD-6)."""
        guard = LockboxGuard(tmp_path / "usage.jsonl")
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "tradestation_crypto_futures",
                date(2021, 1, 1),
                date(2021, 12, 31),
                purpose="discovery",
                arm_id="path_e_crypto_basis_carry",
            )

    @pytest.mark.parametrize("purpose", ["validation", "characterization"])
    def test_out_of_window_purposes_rejected(self, tmp_path: Path, purpose: str) -> None:
        guard = LockboxGuard(tmp_path / "usage.jsonl")
        with pytest.raises(LockboxViolationError):
            guard.assert_period_allowed(
                "tradestation_crypto_futures",
                date(2023, 1, 1),
                date(2023, 12, 31),
                purpose=purpose,
                arm_id="path_e_crypto_basis_carry",
            )

    def test_dev_smoke_is_still_allowed(self, tmp_path: Path) -> None:
        """The depth probe reads as dev_smoke; it must keep working."""
        guard = LockboxGuard(tmp_path / "usage.jsonl")
        guard.assert_period_allowed(
            "tradestation_crypto_futures",
            date(2021, 1, 1),
            date(2024, 12, 31),
            purpose="dev_smoke",
            arm_id="path_e_crypto_basis",
        )
        assert (tmp_path / "usage.jsonl").exists()


class TestDatabentoOptions:
    """The OPRA options dataset is fold-governed independently of equity bars."""

    def test_opra_dataset_has_fce_windows(self) -> None:
        windows = FCE_LOCKBOX_LEDGER_V1.datasets["databento_opra_options"]
        assert windows.discovery == (date(2020, 1, 1), date(2024, 12, 31))
        assert windows.validation == (date(2025, 1, 1), date(2025, 12, 31))
        assert windows.lockbox_start == date(2026, 1, 1)

    def test_options_discovery_read_allowed(self, tmp_path: Path) -> None:
        guard = LockboxGuard(
            usage_log_path=tmp_path / "lockbox_usage_log.jsonl",
            ledger=FCE_LOCKBOX_LEDGER_V1,
        )
        guard.assert_period_allowed(
            "databento_opra_options",
            date(2020, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="path_a_gamma_flow",
        )
        assert read_log(guard)[0]["dataset"] == "databento_opra_options"

    def test_options_lockbox_read_refused(self, tmp_path: Path) -> None:
        guard = LockboxGuard(
            usage_log_path=tmp_path / "lockbox_usage_log.jsonl",
            ledger=FCE_LOCKBOX_LEDGER_V1,
        )
        with pytest.raises(LockboxViolationError, match="program lockbox"):
            guard.assert_period_allowed(
                "databento_opra_options",
                date(2026, 1, 1),
                date(2026, 12, 31),
                purpose="discovery",
                arm_id="path_a_gamma_flow",
            )

    def test_options_validation_read_within_2025_allowed(self, tmp_path: Path) -> None:
        guard = LockboxGuard(
            usage_log_path=tmp_path / "lockbox_usage_log.jsonl",
            ledger=FCE_LOCKBOX_LEDGER_V1,
        )
        guard.assert_period_allowed(
            "databento_opra_options",
            date(2025, 1, 1),
            date(2025, 12, 31),
            purpose="validation",
            arm_id="path_a_gamma_flow",
        )


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

    def test_load_without_purpose_is_refused_and_does_not_log(self, tmp_path: Path) -> None:
        """Inverted from ``test_load_without_purpose_does_not_log``.

        The same undeclared read is exercised; the expectation is the opposite.
        It used to be served silently and leave no ledger entry. It is now
        refused, and the absence of a ledger entry is the correct record of a
        read that never happened rather than of a read nobody logged.
        """
        self._write_bars(tmp_path, "oanda", "EUR_USD", datetime(2020, 6, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)
        with pytest.raises(UndeclaredReadError):
            ds.load("oanda", "EUR_USD", "1m")
        assert not (tmp_path / "lockbox_usage_log.jsonl").exists()

    def test_load_allow_listed_provider_is_served_and_logged_as_ungoverned(
        self, tmp_path: Path
    ) -> None:
        """Inverted from ``test_load_non_campaign_provider_with_purpose_is_not_guarded``.

        The old test pinned the outcome that mattered -- served -- and nothing
        about the record, so a read that wrote no ledger entry satisfied it. It
        also used ``alpaca``, which is *conditionally* mapped, so it did not say
        what it meant. FRED is on the allow-list with a stated reason, and the
        read it makes is now visible in the ledger as ungoverned.
        """
        self._write_bars(tmp_path, "fred", "TB3MS", datetime(2020, 6, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)

        df = ds.load(
            "fred",
            "TB3MS",
            "1m",
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            purpose="discovery",
            arm_id="idea_99",
        )

        assert isinstance(df, pl.DataFrame)
        entry = json.loads((tmp_path / "lockbox_usage_log.jsonl").read_text().splitlines()[0])
        assert entry["dataset"] is None
        assert entry["governed"] is False
        assert entry["provider"] == "fred"
        assert entry["ungoverned_dataset"] == "ungoverned:fred"

    def test_load_unclassified_provider_under_a_research_purpose_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The sibling: nobody classified this provider, so nothing serves it.

        The exemplar was ``binance_us`` until the market-identity review mapped
        it to ``binance_spot``; the assertion is unchanged and the role moved to
        a string that still holds it. A read of a provider outside both the
        ledger mapping and the ungoverned allow-list is refused rather than
        silently served, whatever the string happens to be.
        """
        self._write_bars(
            tmp_path, UNCLASSIFIED_PROVIDER, "BTC_USD", datetime(2026, 2, 1, tzinfo=UTC)
        )
        ds = DataService(data_root=tmp_path)

        with pytest.raises(UnclassifiedProviderError, match=UNCLASSIFIED_PROVIDER):
            ds.load(
                UNCLASSIFIED_PROVIDER,
                "BTC_USD",
                "1m",
                start=date(2026, 1, 1),
                end=date(2026, 12, 31),
                purpose="discovery",
                arm_id="idea_99",
            )
        assert not (tmp_path / "lockbox_usage_log.jsonl").exists()

    def test_the_exemplar_is_genuinely_unclassified(self) -> None:
        """The role these tests give the string must actually be true of it.

        ``binance_us`` held it and stopped being unclassified without the tests
        noticing, which would have left three of them asserting a refusal that
        no longer describes the provider they name. This fails the moment the
        replacement is mapped or allow-listed, which is the signal to move the
        role again rather than to weaken the assertion.
        """
        assert resolve_dataset(UNCLASSIFIED_PROVIDER, "BTC_USD") is None
        assert UNCLASSIFIED_PROVIDER not in UNGOVERNED_PROVIDERS

    def test_binance_us_2026_research_read_is_refused_as_a_program_lockbox_read(
        self, tmp_path: Path
    ) -> None:
        """The change of refusal *type* is the whole point of the mapping.

        Before, this read was refused because nobody had classified the
        provider -- a refusal that says only that the question was open. It is
        now refused by the fold boundary that already governed the venue, and
        the message names it. Asserted by type, not by message text: an
        ``except UnclassifiedProviderError`` upstream must no longer see this.
        """
        self._write_bars(tmp_path, "binance_us", "BTC_USDT", datetime(2026, 2, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)

        with pytest.raises(LockboxViolationError) as excinfo:
            ds.load(
                "binance_us",
                "BTC_USDT",
                "1m",
                start=date(2026, 1, 1),
                end=date(2026, 12, 31),
                purpose="discovery",
                arm_id="idea_99",
            )

        assert not isinstance(excinfo.value, UnclassifiedProviderError)
        assert "binance_spot" in str(excinfo.value)
        assert "program lockbox" in str(excinfo.value)
        assert not (tmp_path / "lockbox_usage_log.jsonl").exists()

    def test_binance_us_dev_smoke_read_is_logged_against_binance_spot(self, tmp_path: Path) -> None:
        """The improvement the mapping buys for the streams that read it.

        ``btc4h_evolution`` and ``regime_stack`` read this surface under
        ``dev_smoke``. That read used to be recorded as ungoverned, under a
        synthetic identity belonging to no ledger. It is now attributed to the
        dataset that governs the venue.
        """
        self._write_bars(tmp_path, "binance_us", "BTC_USDT", datetime(2024, 2, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)

        ds.load(
            "binance_us",
            "BTC_USDT",
            "1m",
            start=date(2024, 1, 1),
            end=date(2024, 12, 31),
            purpose="dev_smoke",
            arm_id="btc4h_evolution",
        )

        entry = json.loads((tmp_path / "lockbox_usage_log.jsonl").read_text().splitlines()[0])
        assert entry["dataset"] == "binance_spot"
        assert "governed" not in entry
        assert "ungoverned_dataset" not in entry

    def test_the_three_refusals_are_distinguishable_types(self) -> None:
        """ "You did not say", "you may not" and "nobody classified this".

        Three different facts. A reader of a traceback, and an ``except``
        clause, must be able to tell them apart without matching message text,
        so none of the three may be a subclass of another.
        """
        assert not issubclass(UnclassifiedProviderError, LockboxViolationError)
        assert not issubclass(UnclassifiedProviderError, UndeclaredReadError)
        assert not issubclass(LockboxViolationError, UnclassifiedProviderError)
        assert not issubclass(UndeclaredReadError, UnclassifiedProviderError)

    def test_load_unclassified_provider_under_dev_smoke_is_served_and_logged(
        self, tmp_path: Path
    ) -> None:
        """Infrastructure keeps working, and stops being invisible.

        Ingestion, fixture builds and diagnostics read unclassified providers
        legitimately. They declare ``dev_smoke``, which is never evidence, and
        the read now leaves a record naming the provider.
        """
        self._write_bars(
            tmp_path, UNCLASSIFIED_PROVIDER, "BTC_USD", datetime(2026, 2, 1, tzinfo=UTC)
        )
        ds = DataService(data_root=tmp_path)

        df = ds.load(
            UNCLASSIFIED_PROVIDER,
            "BTC_USD",
            "1m",
            start=date(2026, 1, 1),
            end=date(2026, 12, 31),
            purpose="dev_smoke",
            arm_id="ingest_probe",
        )

        assert isinstance(df, pl.DataFrame)
        entry = json.loads((tmp_path / "lockbox_usage_log.jsonl").read_text().splitlines()[0])
        assert entry["dataset"] is None
        assert entry["governed"] is False
        assert entry["provider"] == UNCLASSIFIED_PROVIDER
        assert entry["ungoverned_dataset"] == f"ungoverned:{UNCLASSIFIED_PROVIDER}"
        assert entry["purpose"] == "dev_smoke"

    def test_unclassified_research_read_is_refused_before_the_store_is_touched(
        self, tmp_path: Path
    ) -> None:
        """No bars are written, so a served read would raise ``FileNotFoundError``.

        Getting the refusal instead proves the check runs ahead of storage.
        """
        ds = DataService(data_root=tmp_path)
        with pytest.raises(UnclassifiedProviderError):
            ds.load(
                UNCLASSIFIED_PROVIDER,
                "BTC_USD",
                "1m",
                start=date(2020, 1, 1),
                end=date(2020, 12, 31),
                purpose="discovery",
                arm_id="idea_99",
            )

    def test_conditionally_mapped_provider_without_its_discriminator_is_refused(
        self, tmp_path: Path
    ) -> None:
        """``alpaca`` is governed only with ``asset_class='index_recon_book'``.

        A caller who forgets the discriminator used to get a silently ungoverned
        read of a correctly named, mapped provider. It is now unclassified,
        which is the truthful answer: nobody decided what an Alpaca read outside
        the reconstitution book is governed by.
        """
        self._write_bars(tmp_path, "alpaca", "AAPL", datetime(2026, 2, 1, tzinfo=UTC))
        ds = DataService(data_root=tmp_path)

        with pytest.raises(UnclassifiedProviderError, match="alpaca"):
            ds.load(
                "alpaca",
                "AAPL",
                "1m",
                start=date(2026, 1, 1),
                end=date(2026, 12, 31),
                purpose="discovery",
                arm_id="idea_99",
            )

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


class TestGuardedWindows:
    def test_returns_logged_windows_for_arm(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "coinbase_spot",
            date(2020, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="arm_x",
        )
        assert guard.guarded_windows(arm_id="arm_x", dataset="coinbase_spot") == [
            ("2020-01-01", "2024-12-31")
        ]

    def test_filters_by_arm_and_dataset(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "coinbase_spot",
            date(2020, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="arm_x",
        )
        guard.assert_period_allowed(
            "oanda_fx",
            date(2020, 1, 1),
            date(2024, 12, 31),
            purpose="discovery",
            arm_id="arm_y",
        )
        assert guard.guarded_windows(arm_id="arm_x", dataset="oanda_fx") == []
        assert guard.guarded_windows(arm_id="arm_y", dataset="oanda_fx") == [
            ("2020-01-01", "2024-12-31")
        ]

    def test_empty_when_no_reads_logged(self, guard: LockboxGuard) -> None:
        assert guard.guarded_windows(arm_id="nobody", dataset="coinbase_spot") == []

    def test_dev_smoke_window_is_not_citable_coverage(self, guard: LockboxGuard) -> None:
        guard.assert_period_allowed(
            "coinbase_spot",
            date(2026, 1, 1),
            date(2026, 1, 2),
            purpose="dev_smoke",
            arm_id="arm_x",
        )
        assert guard.guarded_windows(arm_id="arm_x", dataset="coinbase_spot") == []

    def test_current_session_only_excludes_prior_runs(self, tmp_path: Path) -> None:
        log_path = tmp_path / "lockbox_usage_log.jsonl"
        first = LockboxGuard(usage_log_path=log_path)
        second = LockboxGuard(usage_log_path=log_path)
        for guard in (first, second):
            guard.assert_period_allowed(
                "coinbase_spot",
                date(2020, 1, 1),
                date(2024, 12, 31),
                purpose="discovery",
                arm_id="arm_x",
            )

        assert len(second.guarded_windows(arm_id="arm_x", dataset="coinbase_spot")) == 2
        assert second.guarded_windows(
            arm_id="arm_x", dataset="coinbase_spot", current_session_only=True
        ) == [("2020-01-01", "2024-12-31")]

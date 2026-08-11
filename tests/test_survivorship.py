"""TDD pins for the point-in-time survivorship audit helpers.

A membership snapshot file records who was in an index on each date; a bar
store records who has data. When the store holds only *current* members, any
long-history cross-sectional study silently conditions on survival — deletions
are the losers, so dropping them is not a partial bias.

These helpers quantify that gap so a window can be chosen, or a bias cap
written, on measured coverage rather than assumption. They are pure functions
over a snapshot frame and an available-symbol set; no I/O.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from liq.data.exceptions import ValidationError
from liq.data.survivorship import (
    SurvivorshipAudit,
    audit_survivorship,
    current_members,
    ever_members,
)


def _snapshots(rows: list[tuple[date, list[str]]]) -> pl.DataFrame:
    """Build a (date, tickers) snapshot frame in the store's schema."""
    return pl.DataFrame(
        {"date": [r[0] for r in rows], "tickers": [r[1] for r in rows]},
        schema={"date": pl.Date, "tickers": pl.List(pl.String)},
    )


# A three-snapshot index where DEAD1 and DEAD2 are deleted over time and
# NEW1 is added, so ever-membership (5) exceeds current membership (3).
_BASE = _snapshots(
    [
        (date(2020, 1, 1), ["AAA", "BBB", "DEAD1"]),
        (date(2021, 1, 1), ["AAA", "BBB", "DEAD2"]),
        (date(2022, 1, 1), ["AAA", "BBB", "NEW1"]),
    ]
)


# ----- ever_members --------------------------------------------------------


class TestEverMembers:
    def test_unions_every_snapshot(self) -> None:
        assert ever_members(_BASE) == {"AAA", "BBB", "DEAD1", "DEAD2", "NEW1"}

    def test_normalizes_case(self) -> None:
        frame = _snapshots([(date(2020, 1, 1), ["aaa", "Bbb"])])
        assert ever_members(frame) == {"AAA", "BBB"}

    def test_window_start_excludes_earlier_snapshots(self) -> None:
        # DEAD1 only ever appears in the 2020 snapshot.
        assert ever_members(_BASE, start=date(2021, 1, 1)) == {
            "AAA",
            "BBB",
            "DEAD2",
            "NEW1",
        }

    def test_window_end_excludes_later_snapshots(self) -> None:
        assert ever_members(_BASE, end=date(2021, 6, 1)) == {
            "AAA",
            "BBB",
            "DEAD1",
            "DEAD2",
        }

    def test_window_bounds_are_inclusive(self) -> None:
        single = ever_members(_BASE, start=date(2021, 1, 1), end=date(2021, 1, 1))
        assert single == {"AAA", "BBB", "DEAD2"}

    def test_empty_snapshots_rejected(self) -> None:
        empty = _snapshots([])
        with pytest.raises(ValidationError, match="no snapshots"):
            ever_members(empty)

    def test_window_selecting_no_snapshots_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no snapshots"):
            ever_members(_BASE, start=date(2030, 1, 1))


# ----- current_members -----------------------------------------------------


class TestCurrentMembers:
    def test_uses_latest_snapshot(self) -> None:
        assert current_members(_BASE) == {"AAA", "BBB", "NEW1"}

    def test_unsorted_input_still_resolves_latest(self) -> None:
        shuffled = _BASE.sort("date", descending=True)
        assert current_members(shuffled) == {"AAA", "BBB", "NEW1"}

    def test_respects_window_end(self) -> None:
        # As of mid-2021 the latest snapshot is the 2021 one.
        assert current_members(_BASE, end=date(2021, 6, 1)) == {"AAA", "BBB", "DEAD2"}


# ----- audit_survivorship --------------------------------------------------


class TestAuditSurvivorship:
    def test_store_with_only_current_members_reports_zero_delisted_coverage(self) -> None:
        audit = audit_survivorship(snapshots=_BASE, available=["AAA", "BBB", "NEW1"])

        assert isinstance(audit, SurvivorshipAudit)
        assert audit.ever_members == 5
        assert audit.current_members == 3
        assert audit.members_with_data == 3
        assert audit.delisted_members == 2
        assert audit.delisted_with_data == 0
        assert audit.delisted_coverage_ratio == 0.0
        assert audit.coverage_ratio == pytest.approx(3 / 5)

    def test_missing_symbols_are_reported_sorted(self) -> None:
        audit = audit_survivorship(snapshots=_BASE, available=["AAA", "BBB", "NEW1"])
        assert audit.missing_symbols == ("DEAD1", "DEAD2")

    def test_partial_delisted_coverage_is_measured(self) -> None:
        audit = audit_survivorship(snapshots=_BASE, available=["AAA", "BBB", "NEW1", "DEAD1"])
        assert audit.delisted_with_data == 1
        assert audit.delisted_coverage_ratio == pytest.approx(0.5)
        assert audit.missing_symbols == ("DEAD2",)

    def test_complete_store_reports_full_coverage(self) -> None:
        audit = audit_survivorship(
            snapshots=_BASE, available=["AAA", "BBB", "NEW1", "DEAD1", "DEAD2"]
        )
        assert audit.coverage_ratio == 1.0
        assert audit.delisted_coverage_ratio == 1.0
        assert audit.missing_symbols == ()

    def test_available_symbols_outside_the_index_are_ignored(self) -> None:
        audit = audit_survivorship(
            snapshots=_BASE, available=["AAA", "BBB", "NEW1", "NOT_IN_INDEX"]
        )
        assert audit.members_with_data == 3
        assert audit.ever_members == 5

    def test_available_symbols_are_case_normalized(self) -> None:
        audit = audit_survivorship(snapshots=_BASE, available=["aaa", "bbb", "new1"])
        assert audit.members_with_data == 3

    def test_no_delisted_members_yields_vacuous_full_ratio(self) -> None:
        stable = _snapshots([(date(2020, 1, 1), ["AAA"]), (date(2021, 1, 1), ["AAA"])])
        audit = audit_survivorship(snapshots=stable, available=["AAA"])
        assert audit.delisted_members == 0
        assert audit.delisted_coverage_ratio == 1.0

    def test_window_narrows_the_audited_universe(self) -> None:
        # Restricting to 2021+ drops DEAD1 from ever-membership entirely.
        audit = audit_survivorship(
            snapshots=_BASE,
            available=["AAA", "BBB", "NEW1"],
            start=date(2021, 1, 1),
        )
        assert audit.ever_members == 4
        assert audit.delisted_members == 1
        assert audit.missing_symbols == ("DEAD2",)

    def test_empty_available_set_is_allowed(self) -> None:
        audit = audit_survivorship(snapshots=_BASE, available=[])
        assert audit.members_with_data == 0
        assert audit.coverage_ratio == 0.0

    def test_audit_is_serializable_for_artifacts(self) -> None:
        audit = audit_survivorship(snapshots=_BASE, available=["AAA", "BBB", "NEW1"])
        payload = audit.to_dict()
        assert payload["ever_members"] == 5
        assert payload["delisted_with_data"] == 0
        assert payload["missing_symbols"] == ["DEAD1", "DEAD2"]

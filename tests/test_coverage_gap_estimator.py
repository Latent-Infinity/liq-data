"""TDD pins for the gap-aware coverage + cost estimator helpers.

The helpers are pure functions that compose CoverageManifest and a
cost-estimate callable; they live in ``liq.data.coverage_gap_estimator``.
The CLI under ``liq-experiments/scripts`` wires them to
``DataService.estimate_databento_cost`` for the live API call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from liq.data.coverage_gap_estimator import (
    GapPerSymbol,
    coverage_summary,
    group_symbols_by_gap_pattern,
    per_symbol_gaps,
)
from liq.data.manifest import CoverageManifest, CoverageRange


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _write_manifest(
    root: Path,
    *,
    provider: str,
    dataset: str,
    timeframe: str,
    symbol: str,
    ranges: list[tuple[datetime, datetime]],
) -> None:
    """Helper: build and flush a CoverageManifest under ``root``."""
    cov = CoverageManifest.load(
        root=root, provider=provider, dataset=dataset, timeframe=timeframe, symbol=symbol
    )
    for start, end in ranges:
        cov.record(CoverageRange(start=start, end=end, fetched_at=_utc(2026, 6, 27)))
    cov.flush()


# ----- per_symbol_gaps -----------------------------------------------------


class TestPerSymbolGaps:
    def test_fully_covered_symbol_has_empty_gaps(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
            ranges=[(_utc(2024, 1, 1), _utc(2025, 1, 1))],
        )
        gaps = per_symbol_gaps(
            symbols=["AAPL"],
            data_root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            start=_utc(2024, 1, 1),
            end=_utc(2025, 1, 1),
        )
        assert gaps["AAPL"] == []

    def test_uncovered_symbol_has_one_full_gap(self, tmp_path: Path) -> None:
        # No manifest written = no coverage
        gaps = per_symbol_gaps(
            symbols=["MISSING"],
            data_root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            start=_utc(2024, 1, 1),
            end=_utc(2025, 1, 1),
        )
        assert gaps["MISSING"] == [(_utc(2024, 1, 1), _utc(2025, 1, 1))]

    def test_partial_coverage_returns_remaining_gaps(self, tmp_path: Path) -> None:
        # Covered only [2024-04 .. 2024-08); want [2024-01 .. 2025-01)
        _write_manifest(
            tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="X",
            ranges=[(_utc(2024, 4, 1), _utc(2024, 8, 1))],
        )
        gaps = per_symbol_gaps(
            symbols=["X"],
            data_root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            start=_utc(2024, 1, 1),
            end=_utc(2025, 1, 1),
        )
        assert gaps["X"] == [
            (_utc(2024, 1, 1), _utc(2024, 4, 1)),
            (_utc(2024, 8, 1), _utc(2025, 1, 1)),
        ]

    def test_multiple_symbols_returned_in_input_order(self, tmp_path: Path) -> None:
        _write_manifest(
            tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="A",
            ranges=[(_utc(2024, 1, 1), _utc(2025, 1, 1))],
        )
        gaps = per_symbol_gaps(
            symbols=["A", "B"],
            data_root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            start=_utc(2024, 1, 1),
            end=_utc(2025, 1, 1),
        )
        assert list(gaps) == ["A", "B"]
        assert gaps["A"] == []
        assert gaps["B"] == [(_utc(2024, 1, 1), _utc(2025, 1, 1))]


# ----- group_symbols_by_gap_pattern ---------------------------------------


class TestGroupByGapPattern:
    def test_symbols_with_identical_gaps_grouped_together(self) -> None:
        gaps: GapPerSymbol = {
            "A": [(_utc(2023, 1, 1), _utc(2024, 1, 1))],
            "B": [(_utc(2023, 1, 1), _utc(2024, 1, 1))],
            "C": [(_utc(2024, 6, 1), _utc(2025, 1, 1))],
        }
        groups = group_symbols_by_gap_pattern(gaps)
        # Two distinct patterns: AB and C
        assert len(groups) == 2
        symbols_in_each = [sorted(s) for _, s in groups]
        assert sorted(symbols_in_each) == [["A", "B"], ["C"]]

    def test_fully_covered_symbols_grouped_into_empty_bucket(self) -> None:
        gaps: GapPerSymbol = {"A": [], "B": [], "C": [(_utc(2023, 1, 1), _utc(2024, 1, 1))]}
        groups = group_symbols_by_gap_pattern(gaps)
        # The empty-gap bucket is included so the caller can identify
        # already-covered symbols.
        empty_buckets = [g for g in groups if not g[0]]
        assert len(empty_buckets) == 1
        assert sorted(empty_buckets[0][1]) == ["A", "B"]


# ----- coverage_summary ---------------------------------------------------


class TestCoverageSummary:
    def test_summary_counts_symbol_categories(self) -> None:
        gaps: GapPerSymbol = {
            "FULL_A": [],
            "FULL_B": [],
            "PARTIAL": [(_utc(2024, 1, 1), _utc(2024, 4, 1))],
            "MISSING_A": [(_utc(2024, 1, 1), _utc(2025, 1, 1))],
            "MISSING_B": [(_utc(2024, 1, 1), _utc(2025, 1, 1))],
        }
        s = coverage_summary(
            gaps,
            target_start=_utc(2024, 1, 1),
            target_end=_utc(2025, 1, 1),
        )
        assert s["n_symbols_total"] == 5
        assert s["n_fully_covered"] == 2
        assert s["n_partial"] == 1
        assert s["n_missing"] == 2

    def test_summary_reports_total_gap_symbol_days(self) -> None:
        gaps: GapPerSymbol = {
            "A": [(_utc(2024, 1, 1), _utc(2024, 1, 11))],  # 10 days
            "B": [(_utc(2024, 1, 1), _utc(2024, 1, 6))],  # 5 days
        }
        s = coverage_summary(
            gaps,
            target_start=_utc(2024, 1, 1),
            target_end=_utc(2024, 12, 31),
        )
        assert s["total_gap_symbol_days"] == 15


# ----- timezone tolerance --------------------------------------------------


def test_naive_datetimes_are_treated_as_utc(tmp_path: Path) -> None:
    """The caller may pass naive datetimes; the helpers should treat them
    as UTC rather than crashing on tz comparison.
    """
    _write_manifest(
        tmp_path,
        provider="databento",
        dataset="EQUS.MINI",
        timeframe="1m",
        symbol="AAPL",
        ranges=[(_utc(2024, 1, 1), _utc(2025, 1, 1))],
    )
    gaps = per_symbol_gaps(
        symbols=["AAPL"],
        data_root=tmp_path,
        provider="databento",
        dataset="EQUS.MINI",
        timeframe="1m",
        start=datetime(2024, 1, 1),  # naive
        end=datetime(2025, 1, 1),  # naive
    )
    assert gaps["AAPL"] == []

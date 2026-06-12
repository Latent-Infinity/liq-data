"""TDD tests for the coverage manifest.

Locked contract:

* One manifest per (provider, dataset, timeframe, symbol).
* Storage layout: ``metadata/{provider}/{dataset}/{timeframe}/{symbol}/manifest.parquet``.
* Rows are contiguous ranges; ``record`` merges adjacent/overlapping
  ranges so we never accumulate fragments.
* ``gaps(start, end)`` returns the uncovered slices, *honoring an empty
  manifest as one big gap*.
* ``transaction()`` is the only safe way to update — body runs inside;
  raising rolls back to the pre-transaction state. Successful body
  commits in one write.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from liq.data.manifest import CoverageManifest, CoverageRange, ManifestRollbackError


def _ts(day: int) -> datetime:
    return datetime(2025, 1, day, tzinfo=UTC)


# ----- range merging --------------------------------------------------------


class TestCoverageRangeMerge:
    def test_disjoint_ranges_stay_separate(self) -> None:
        ranges = CoverageManifest.merge_ranges(
            [
                CoverageRange(start=_ts(1), end=_ts(2), fetched_at=_ts(10)),
                CoverageRange(start=_ts(5), end=_ts(6), fetched_at=_ts(10)),
            ]
        )
        assert len(ranges) == 2

    def test_overlapping_ranges_merge(self) -> None:
        ranges = CoverageManifest.merge_ranges(
            [
                CoverageRange(start=_ts(1), end=_ts(5), fetched_at=_ts(10)),
                CoverageRange(start=_ts(4), end=_ts(7), fetched_at=_ts(11)),
            ]
        )
        assert len(ranges) == 1
        assert ranges[0].start == _ts(1)
        assert ranges[0].end == _ts(7)

    def test_adjacent_ranges_merge(self) -> None:
        ranges = CoverageManifest.merge_ranges(
            [
                CoverageRange(start=_ts(1), end=_ts(5), fetched_at=_ts(10)),
                CoverageRange(start=_ts(5), end=_ts(8), fetched_at=_ts(11)),
            ]
        )
        assert len(ranges) == 1
        assert ranges[0].end == _ts(8)


# ----- load / save round-trips ---------------------------------------------


class TestManifestPersistence:
    def test_empty_manifest_for_missing_file(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        assert m.ranges == []

    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        m.record(CoverageRange(start=_ts(1), end=_ts(2), fetched_at=_ts(10)))
        m.flush()
        m2 = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        assert m2.ranges == m.ranges

    def test_persisted_path_matches_layout_contract(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        assert m.path.relative_to(tmp_path) == Path(
            "metadata/databento/EQUS.MINI/1m/AAPL/manifest.parquet"
        )

    def test_persisted_schema_uses_timestamp_column_names(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        m.record(CoverageRange(start=_ts(1), end=_ts(2), fetched_at=_ts(10)))
        m.flush()
        df = pl.read_parquet(m.path)
        assert df.columns == ["start_ts", "end_ts", "fetched_at", "batch_job_id"]


# ----- gap detection -------------------------------------------------------


class TestManifestGaps:
    def _seeded(self, tmp_path: Path) -> CoverageManifest:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        m.record(CoverageRange(start=_ts(2), end=_ts(4), fetched_at=_ts(10)))
        m.record(CoverageRange(start=_ts(7), end=_ts(9), fetched_at=_ts(11)))
        return m

    def test_empty_manifest_is_one_big_gap(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        gaps = m.gaps(start=_ts(1), end=_ts(10))
        assert gaps == [(_ts(1), _ts(10))]

    def test_no_gaps_when_fully_covered(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        m.record(CoverageRange(start=_ts(1), end=_ts(10), fetched_at=_ts(10)))
        assert m.gaps(start=_ts(2), end=_ts(5)) == []

    def test_gap_before_existing(self, tmp_path: Path) -> None:
        m = self._seeded(tmp_path)
        assert m.gaps(start=_ts(1), end=_ts(3)) == [(_ts(1), _ts(2))]

    def test_gap_between_existing_ranges(self, tmp_path: Path) -> None:
        m = self._seeded(tmp_path)
        assert m.gaps(start=_ts(1), end=_ts(10)) == [
            (_ts(1), _ts(2)),
            (_ts(4), _ts(7)),
            (_ts(9), _ts(10)),
        ]

    def test_gap_after_existing(self, tmp_path: Path) -> None:
        m = self._seeded(tmp_path)
        assert m.gaps(start=_ts(8), end=_ts(12)) == [(_ts(9), _ts(12))]

    def test_request_inside_existing_range_no_gap(self, tmp_path: Path) -> None:
        m = self._seeded(tmp_path)
        assert m.gaps(start=_ts(2), end=_ts(3)) == []

    def test_gaps_validate_window(self, tmp_path: Path) -> None:
        m = self._seeded(tmp_path)
        with pytest.raises(ValueError):
            m.gaps(start=_ts(5), end=_ts(3))


# ----- transaction --------------------------------------------------------


class TestManifestTransaction:
    def test_successful_transaction_commits(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        with m.transaction() as txn:
            txn.record(CoverageRange(start=_ts(1), end=_ts(3), fetched_at=_ts(10)))
        # Persisted to disk on commit.
        m2 = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        assert m2.ranges == m.ranges
        assert m.ranges == [CoverageRange(start=_ts(1), end=_ts(3), fetched_at=_ts(10))]

    def test_transaction_rollback_on_exception(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        m.record(CoverageRange(start=_ts(1), end=_ts(2), fetched_at=_ts(10)))
        m.flush()
        try:
            with m.transaction() as txn:
                txn.record(CoverageRange(start=_ts(5), end=_ts(8), fetched_at=_ts(11)))
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        m2 = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        # Only the pre-transaction state survives.
        assert m2.ranges == [CoverageRange(start=_ts(1), end=_ts(2), fetched_at=_ts(10))]
        assert m.ranges == m2.ranges

    def test_transaction_cannot_be_reused(self, tmp_path: Path) -> None:
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        with m.transaction() as txn:
            txn.record(CoverageRange(start=_ts(1), end=_ts(2), fetched_at=_ts(10)))
        with pytest.raises(ManifestRollbackError), txn:
            pass  # double-enter the same transaction object

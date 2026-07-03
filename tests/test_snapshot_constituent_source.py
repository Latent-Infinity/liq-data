"""Point-in-time constituent resolution from membership snapshots."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from liq.data.universes import SnapshotConstituentSource, UniverseResolutionError


@pytest.fixture
def snapshots() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2005, 1, 3), date(2005, 2, 1), date(2005, 3, 1)],
            "tickers": [["AAA", "BBB", "CCC"], ["AAA", "BBB", "DDD"], ["AAA", "EEE", "DDD"]],
        }
    )


class TestSnapshotConstituentSource:
    def test_is_pit(self, snapshots: pl.DataFrame) -> None:
        source = SnapshotConstituentSource("sp500", snapshots)
        assert source.pit is True

    def test_members_floor_semantics(self, snapshots: pl.DataFrame) -> None:
        source = SnapshotConstituentSource("sp500", snapshots)
        # Exactly on a snapshot date.
        assert source.members(id="sp500", as_of=date(2005, 2, 1)) == ["AAA", "BBB", "DDD"]
        # Between snapshots → most recent snapshot at or before as_of.
        assert source.members(id="sp500", as_of=date(2005, 2, 15)) == ["AAA", "BBB", "DDD"]
        # After the last snapshot → the last snapshot.
        assert source.members(id="sp500", as_of=date(2006, 1, 1)) == ["AAA", "DDD", "EEE"]

    def test_before_first_snapshot_rejected(self, snapshots: pl.DataFrame) -> None:
        source = SnapshotConstituentSource("sp500", snapshots)
        with pytest.raises(UniverseResolutionError, match="before first snapshot"):
            source.members(id="sp500", as_of=date(2004, 12, 31))

    def test_unknown_id_rejected(self, snapshots: pl.DataFrame) -> None:
        source = SnapshotConstituentSource("sp500", snapshots)
        with pytest.raises(UniverseResolutionError, match="unknown constituent id"):
            source.members(id="nasdaq100", as_of=date(2005, 2, 1))

    def test_symbols_returned_sorted_upper(self, snapshots: pl.DataFrame) -> None:
        source = SnapshotConstituentSource(
            "sp500",
            pl.DataFrame({"date": [date(2005, 1, 3)], "tickers": [["bbb", "AAA"]]}),
        )
        assert source.members(id="sp500", as_of=date(2005, 1, 3)) == ["AAA", "BBB"]

    def test_from_parquet_round_trip(self, snapshots: pl.DataFrame, tmp_path: Path) -> None:
        path = tmp_path / "snapshots.parquet"
        snapshots.write_parquet(path)
        source = SnapshotConstituentSource.from_parquet("sp500", path)
        assert source.members(id="sp500", as_of=date(2005, 3, 1)) == ["AAA", "DDD", "EEE"]

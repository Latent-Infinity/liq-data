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

    def test_empty_snapshots_rejected(self) -> None:
        empty = pl.DataFrame(schema={"date": pl.Date, "tickers": pl.List(pl.String)})
        with pytest.raises(UniverseResolutionError, match="must not be empty"):
            SnapshotConstituentSource("sp500", empty)

    def test_missing_snapshot_columns_rejected(self) -> None:
        with pytest.raises(UniverseResolutionError, match="date.*tickers"):
            SnapshotConstituentSource("sp500", pl.DataFrame({"date": [date(2005, 1, 3)]}))

    def test_duplicate_snapshot_dates_rejected(self) -> None:
        duplicate = pl.DataFrame(
            {
                "date": [date(2005, 1, 3), date(2005, 1, 3)],
                "tickers": [["AAA"], ["BBB"]],
            }
        )
        with pytest.raises(UniverseResolutionError, match="duplicate"):
            SnapshotConstituentSource("sp500", duplicate)

    @pytest.mark.parametrize(
        ("snapshots", "message"),
        [
            (
                pl.DataFrame({"date": ["2005-01-03"], "tickers": [["AAA"]]}),
                "date column",
            ),
            (
                pl.DataFrame({"date": [date(2005, 1, 3)], "tickers": ["AAA"]}),
                "tickers column",
            ),
        ],
    )
    def test_snapshot_schema_must_use_dates_and_ticker_lists(
        self, snapshots: pl.DataFrame, message: str
    ) -> None:
        with pytest.raises(UniverseResolutionError, match=message):
            SnapshotConstituentSource("sp500", snapshots)


class TestDirectorySnapshotSource:
    def test_is_pit(self, tmp_path: Path) -> None:
        from liq.data.universes import DirectorySnapshotSource

        assert DirectorySnapshotSource(tmp_path).pit is True

    def test_resolves_per_id_from_directory(self, snapshots: pl.DataFrame, tmp_path: Path) -> None:
        from liq.data.universes import DirectorySnapshotSource

        snapshots.write_parquet(tmp_path / "sp500.parquet")
        source = DirectorySnapshotSource(tmp_path)
        assert source.members(id="sp500", as_of=date(2005, 2, 15)) == ["AAA", "BBB", "DDD"]

    def test_caches_source_per_id(self, snapshots: pl.DataFrame, tmp_path: Path) -> None:
        from liq.data.universes import DirectorySnapshotSource

        path = tmp_path / "sp500.parquet"
        snapshots.write_parquet(path)
        source = DirectorySnapshotSource(tmp_path)
        first = source.members(id="sp500", as_of=date(2006, 1, 1))
        # Overwrite the file; a cached per-id source must ignore the change.
        pl.DataFrame({"date": [date(2005, 1, 3)], "tickers": [["ZZZ"]]}).write_parquet(path)
        assert source.members(id="sp500", as_of=date(2006, 1, 1)) == first == ["AAA", "DDD", "EEE"]

    def test_missing_snapshot_file_is_fail_closed(self, tmp_path: Path) -> None:
        from liq.data.universes import DirectorySnapshotSource, UniverseResolutionError

        source = DirectorySnapshotSource(tmp_path)
        with pytest.raises(UniverseResolutionError, match="no point-in-time membership snapshot"):
            source.members(id="sp500", as_of=date(2005, 2, 1))

    def test_constituent_id_cannot_escape_snapshot_directory(self, tmp_path: Path) -> None:
        from liq.data.universes import DirectorySnapshotSource, UniverseResolutionError

        source = DirectorySnapshotSource(tmp_path / "snapshots")
        with pytest.raises(UniverseResolutionError, match="invalid constituent id"):
            source.members(id="../outside", as_of=date(2005, 2, 1))

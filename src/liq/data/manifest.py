"""Coverage manifest — per-symbol record of what data has been ingested.

The manifest is the idempotency mechanism for ``DataService.sync(...)``:
before any provider call we ask "which ranges of (provider, dataset,
timeframe, symbol) have we already fetched?" and skip them. The
alternative — inspecting stored bars — is ambiguous because a missing
minute could be "not fetched" *or* "no trades occurred" (thin symbols,
halts, partial sessions).

Storage layout (one parquet per symbol/timeframe/dataset):

    metadata/{provider}/{dataset}/{timeframe}/{symbol}/manifest.parquet

Each row is one contiguous covered range with ``start_ts`` / ``end_ts``
columns. ``record(...)`` merges overlapping/adjacent ranges so we never
accumulate fragments. Updates go through
:meth:`CoverageManifest.transaction` — a context manager that snapshots
state on entry and rolls back to that snapshot if the body raises, so
partial writes can't leave the manifest claiming data that didn't
actually land.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

_MANIFEST_FILENAME = "manifest.parquet"


# ----- exceptions -----------------------------------------------------------


class ManifestError(Exception):
    """Base class for manifest failures."""


class ManifestRollbackError(ManifestError):
    """A transaction was misused (double-enter, etc.)."""


# ----- range record ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageRange:
    """One contiguous covered range. Half-open: ``[start, end)``.

    ``end <= start`` is illegal — empty ranges add no information and
    would only confuse the merge logic.
    """

    start: datetime
    end: datetime
    fetched_at: datetime
    batch_job_id: str | None = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"CoverageRange end ({self.end}) must be > start ({self.start})")


# ----- manifest --------------------------------------------------------------


class CoverageManifest:
    """In-memory + parquet-backed list of contiguous coverage ranges.

    Construct via :meth:`load` (the factory ensures the right path
    layout and loads any existing parquet). After ``record(...)`` calls,
    persistence is explicit via :meth:`flush` *or* implicit at the end
    of a successful :meth:`transaction`.
    """

    SCHEMA: dict[str, Any] = {
        "start_ts": pl.Datetime("us", "UTC"),
        "end_ts": pl.Datetime("us", "UTC"),
        "fetched_at": pl.Datetime("us", "UTC"),
        "batch_job_id": pl.Utf8,
    }

    # ----- construction ----------------------------------------------

    def __init__(
        self,
        *,
        path: Path,
        ranges: list[CoverageRange] | None = None,
    ) -> None:
        self._path = path
        self._ranges: list[CoverageRange] = list(ranges or [])

    @classmethod
    def load(
        cls,
        *,
        root: str | Path,
        provider: str,
        dataset: str,
        timeframe: str,
        symbol: str,
    ) -> CoverageManifest:
        """Build a manifest, hydrating from disk if a parquet exists."""
        path = (
            Path(root) / "metadata" / provider / dataset / timeframe / symbol / _MANIFEST_FILENAME
        )
        ranges = cls._read_parquet(path) if path.exists() else []
        return cls(path=path, ranges=ranges)

    # ----- public accessors ------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def ranges(self) -> list[CoverageRange]:
        return list(self._ranges)

    # ----- mutation ---------------------------------------------------

    def record(self, new_range: CoverageRange) -> None:
        """Add ``new_range`` to the manifest, merging adjacent/overlapping rows.

        The merged list is kept sorted by ``start``.
        """
        merged = self.merge_ranges([*self._ranges, new_range])
        self._ranges = merged

    def flush(self) -> None:
        """Write the current in-memory state to ``self.path`` atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        df = self._ranges_to_dataframe(self._ranges)
        tmp = self._path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp)
        tmp.replace(self._path)

    # ----- gap detection ---------------------------------------------

    def gaps(self, *, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Uncovered slices of ``[start, end)`` not already in the manifest.

        Returns a sorted list of ``(gap_start, gap_end)`` pairs. Empty
        list means full coverage; ``[(start, end)]`` means empty
        manifest.
        """
        if end <= start:
            raise ValueError(f"gap window end ({end}) must be > start ({start})")

        cursor = start
        gaps: list[tuple[datetime, datetime]] = []
        for r in self._ranges:
            if r.end <= cursor:
                continue
            if r.start >= end:
                break
            if r.start > cursor:
                gaps.append((cursor, min(r.start, end)))
            cursor = max(cursor, r.end)
            if cursor >= end:
                break
        if cursor < end:
            gaps.append((cursor, end))
        return gaps

    # ----- transaction -----------------------------------------------

    def transaction(self) -> _ManifestTransaction:
        """Context manager that snapshots state and rolls back on error.

        Successful body commits the merged state to disk in a single
        flush. Multiple ``record(...)`` calls in one transaction
        therefore make at most one parquet write.
        """
        return _ManifestTransaction(self)

    # ----- helpers ---------------------------------------------------

    @staticmethod
    def merge_ranges(ranges: list[CoverageRange]) -> list[CoverageRange]:
        """Sort + merge overlapping or adjacent ranges.

        ``fetched_at`` of the merged row is the *latest* contributor's
        ``fetched_at``; ``batch_job_id`` survives if and only if every
        merged contributor had the same id (otherwise dropped to
        ``None`` — mixed-source ranges have no canonical job id).
        """
        if not ranges:
            return []
        sorted_ranges = sorted(ranges, key=lambda r: r.start)
        out: list[CoverageRange] = [sorted_ranges[0]]
        for r in sorted_ranges[1:]:
            top = out[-1]
            if r.start <= top.end:
                merged_end = max(top.end, r.end)
                merged_fetched = max(top.fetched_at, r.fetched_at)
                merged_job = top.batch_job_id if top.batch_job_id == r.batch_job_id else None
                out[-1] = CoverageRange(
                    start=top.start,
                    end=merged_end,
                    fetched_at=merged_fetched,
                    batch_job_id=merged_job,
                )
            else:
                out.append(r)
        return out

    @classmethod
    def _ranges_to_dataframe(cls, ranges: list[CoverageRange]) -> pl.DataFrame:
        if not ranges:
            return pl.DataFrame(schema=cls.SCHEMA)
        rows = [
            {
                "start_ts": r.start,
                "end_ts": r.end,
                "fetched_at": r.fetched_at,
                "batch_job_id": r.batch_job_id,
            }
            for r in ranges
        ]
        return pl.DataFrame(rows, schema=cls.SCHEMA)

    @classmethod
    def _read_parquet(cls, path: Path) -> list[CoverageRange]:
        df = pl.read_parquet(path)
        start_col = "start_ts" if "start_ts" in df.columns else "start"
        end_col = "end_ts" if "end_ts" in df.columns else "end"
        out: list[CoverageRange] = []
        for row in df.iter_rows(named=True):
            out.append(
                CoverageRange(
                    start=row[start_col],
                    end=row[end_col],
                    fetched_at=row["fetched_at"],
                    batch_job_id=row["batch_job_id"],
                )
            )
        return out


# ----- transaction --------------------------------------------------------


class _ManifestTransaction:
    """Internal context manager — see :meth:`CoverageManifest.transaction`."""

    def __init__(self, manifest: CoverageManifest) -> None:
        self._manifest = manifest
        self._snapshot: list[CoverageRange] | None = None
        self._consumed = False

    def __enter__(self) -> _ManifestTransaction:
        if self._consumed:
            raise ManifestRollbackError("manifest transaction object cannot be reused")
        self._snapshot = list(self._manifest._ranges)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._consumed = True
        if exc is not None:
            # Roll back to the snapshot; never reach disk.
            assert self._snapshot is not None
            self._manifest._ranges = self._snapshot
            return
        # Successful body → commit.
        self._manifest.flush()

    def record(self, new_range: CoverageRange) -> None:
        self._manifest.record(new_range)


__all__ = [
    "CoverageManifest",
    "CoverageRange",
    "ManifestError",
    "ManifestRollbackError",
]

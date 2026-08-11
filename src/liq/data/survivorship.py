"""Point-in-time survivorship audit for index-membership studies.

A membership snapshot file records the full index composition on each
snapshot date. A bar store records which symbols have data. When the store
holds only *current* members, a long-history cross-sectional study silently
conditions on survival: index deletions are, on average, the losers, so
dropping them is not a partial bias that a wider null can absorb.

These helpers measure the gap — how many names were ever members, how many of
those have data, and specifically how many *deleted* names have data — so a
study window can be chosen, or a bias cap written, against a measured number.

The functions are pure over a snapshot frame and an available-symbol
collection; file loading stays with
:class:`liq.data.universes.SnapshotConstituentSource`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from liq.data.exceptions import ValidationError

__all__ = [
    "SurvivorshipAudit",
    "audit_survivorship",
    "current_members",
    "ever_members",
]


def _windowed(snapshots: pl.DataFrame, start: date | None, end: date | None) -> pl.DataFrame:
    """Sort by date and restrict to the inclusive ``[start, end]`` window."""
    frame = snapshots.sort("date")
    if start is not None:
        frame = frame.filter(pl.col("date") >= start)
    if end is not None:
        frame = frame.filter(pl.col("date") <= end)
    if frame.height == 0:
        raise ValidationError(
            f"no snapshots in window start={start} end={end}; "
            "cannot audit survivorship over an empty membership history"
        )
    return frame


def _normalize(symbols: Iterable[str]) -> set[str]:
    return {str(s).upper() for s in symbols}


def _ratio(numerator: int, denominator: int) -> float:
    """Coverage ratio; an empty denominator is vacuously complete."""
    if denominator == 0:
        return 1.0
    return numerator / denominator


def ever_members(
    snapshots: pl.DataFrame,
    *,
    start: date | None = None,
    end: date | None = None,
) -> set[str]:
    """Every symbol that appears in any snapshot within the window.

    Args:
        snapshots: Frame with a ``date`` column and a ``tickers`` list column.
        start: Inclusive lower bound on snapshot date.
        end: Inclusive upper bound on snapshot date.

    Raises:
        ValidationError: The window selects no snapshots.
    """
    frame = _windowed(snapshots, start, end)
    members: set[str] = set()
    for row in frame["tickers"].to_list():
        members |= _normalize(row)
    return members


def current_members(
    snapshots: pl.DataFrame,
    *,
    end: date | None = None,
) -> set[str]:
    """Membership as of the latest snapshot at or before ``end``.

    Raises:
        ValidationError: The window selects no snapshots.
    """
    frame = _windowed(snapshots, None, end)
    return _normalize(frame["tickers"].to_list()[-1])


@dataclass(frozen=True)
class SurvivorshipAudit:
    """Measured survivorship coverage of a bar store against a PIT index.

    ``delisted_*`` fields carry the decisive numbers: a store whose
    ``delisted_coverage_ratio`` is 0.0 can support no unbiased long-history
    cross-sectional claim, regardless of how deep its bars run.
    """

    ever_members: int
    current_members: int
    members_with_data: int
    delisted_members: int
    delisted_with_data: int
    coverage_ratio: float
    delisted_coverage_ratio: float
    missing_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for verification artifacts."""
        return {
            "ever_members": self.ever_members,
            "current_members": self.current_members,
            "members_with_data": self.members_with_data,
            "delisted_members": self.delisted_members,
            "delisted_with_data": self.delisted_with_data,
            "coverage_ratio": self.coverage_ratio,
            "delisted_coverage_ratio": self.delisted_coverage_ratio,
            "missing_symbols": list(self.missing_symbols),
        }


def audit_survivorship(
    *,
    snapshots: pl.DataFrame,
    available: Iterable[str],
    start: date | None = None,
    end: date | None = None,
) -> SurvivorshipAudit:
    """Compare PIT index membership against the symbols a store actually has.

    Args:
        snapshots: Frame with a ``date`` column and a ``tickers`` list column.
        available: Symbols for which the store holds data. Symbols outside the
            index are ignored, so a raw store listing can be passed directly.
        start: Inclusive lower bound on snapshot date.
        end: Inclusive upper bound on snapshot date.

    Raises:
        ValidationError: The window selects no snapshots.
    """
    ever = ever_members(snapshots, start=start, end=end)
    current = current_members(snapshots, end=end)
    have = _normalize(available)

    delisted = ever - current
    missing = ever - have

    return SurvivorshipAudit(
        ever_members=len(ever),
        current_members=len(current),
        members_with_data=len(ever & have),
        delisted_members=len(delisted),
        delisted_with_data=len(delisted & have),
        coverage_ratio=_ratio(len(ever & have), len(ever)),
        delisted_coverage_ratio=_ratio(len(delisted & have), len(delisted)),
        missing_symbols=tuple(sorted(missing)),
    )

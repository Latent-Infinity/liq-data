"""Gap-aware coverage analysis for cost-estimating fetches.

For a target ``(universe, dataset, timeframe, start, end)``, these
helpers compose :class:`liq.data.manifest.CoverageManifest` into:

* :func:`per_symbol_gaps` — the missing time ranges per symbol over the
  target window.
* :func:`group_symbols_by_gap_pattern` — buckets symbols whose gap
  signatures are identical, so a single batched cost estimate can cover
  all symbols in a bucket.
* :func:`coverage_summary` — aggregate statistics (fully covered,
  partial, missing counts + total gap symbol-days) for human display.

The cost call itself lives in :meth:`DataService.estimate_databento_cost`;
callers wire it to ``per_symbol_gaps`` output. This module is
intentionally I/O-light (only ``CoverageManifest.load`` per symbol).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

from liq.data.manifest import CoverageManifest

# Per-symbol gap dict: {symbol: [(start, end), ...]} in chronological order.
GapPerSymbol: TypeAlias = dict[str, list[tuple[datetime, datetime]]]


def _to_utc(ts: datetime) -> datetime:
    """Treat naive datetimes as UTC; pass tz-aware datetimes through."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def per_symbol_gaps(
    *,
    symbols: list[str],
    data_root: Path,
    provider: str,
    dataset: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> GapPerSymbol:
    """For each symbol, return the missing time ranges within ``[start, end)``.

    A symbol with no manifest on disk is treated as fully uncovered (one
    gap matching the full target window). Existing manifests are consulted
    via :meth:`CoverageManifest.gaps` — the canonical gap-computation
    surface.

    Returns a dict keyed by ``symbols`` in input order; values are
    chronological lists of ``(gap_start, gap_end)`` half-open intervals.
    Empty list means fully covered.
    """
    s = _to_utc(start)
    e = _to_utc(end)
    out: GapPerSymbol = {}
    for symbol in symbols:
        manifest = CoverageManifest.load(
            root=data_root,
            provider=provider,
            dataset=dataset,
            timeframe=timeframe,
            symbol=symbol,
        )
        out[symbol] = list(manifest.gaps(start=s, end=e))
    return out


def group_symbols_by_gap_pattern(
    gaps: GapPerSymbol,
) -> list[tuple[list[tuple[datetime, datetime]], list[str]]]:
    """Bucket symbols whose gap lists are identical.

    Returns a list of ``(gap_pattern, symbols)`` tuples. The empty-gap
    bucket is included if any symbols are fully covered, so callers can
    identify "no-fetch-needed" symbols.
    """
    buckets: dict[tuple[tuple[datetime, datetime], ...], list[str]] = {}
    for symbol, sym_gaps in gaps.items():
        key = tuple(sym_gaps)
        buckets.setdefault(key, []).append(symbol)
    return [(list(pattern), symbols) for pattern, symbols in buckets.items()]


def coverage_summary(
    gaps: GapPerSymbol,
    *,
    target_start: datetime,
    target_end: datetime,
) -> dict[str, object]:
    """Aggregate statistics over the per-symbol gap dict.

    Categories:
    * ``n_fully_covered`` — symbols with no gap in ``[target_start, target_end)``.
    * ``n_partial`` — symbols with at least one gap shorter than the full window.
    * ``n_missing`` — symbols whose single gap equals the full window.
    """
    ts = _to_utc(target_start)
    te = _to_utc(target_end)
    full_window = (ts, te)

    n_full = 0
    n_partial = 0
    n_missing = 0
    total_gap_days = 0.0
    for sym_gaps in gaps.values():
        if not sym_gaps:
            n_full += 1
            continue
        if len(sym_gaps) == 1 and sym_gaps[0] == full_window:
            n_missing += 1
        else:
            n_partial += 1
        for g_start, g_end in sym_gaps:
            total_gap_days += (g_end - g_start).total_seconds() / 86400.0

    return {
        "n_symbols_total": len(gaps),
        "n_fully_covered": n_full,
        "n_partial": n_partial,
        "n_missing": n_missing,
        "total_gap_symbol_days": int(round(total_gap_days)),
        "target_start": ts.isoformat(),
        "target_end": te.isoformat(),
    }


__all__ = [
    "GapPerSymbol",
    "coverage_summary",
    "group_symbols_by_gap_pattern",
    "per_symbol_gaps",
]

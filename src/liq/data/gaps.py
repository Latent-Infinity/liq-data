"""Gap detection and handling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, datetime, UTC

import polars as pl


@dataclass
class GapPolicy:
    """Policy for gap handling per market/provider."""

    expected_gap_minutes: int = 0
    max_forward_fill_minutes: int = 0
    mark_weekends: bool = False


def classify_gaps(df: pl.DataFrame, policy: GapPolicy) -> pl.DataFrame:
    """Annotate gaps in a timestamp-sorted DataFrame."""
    if df.is_empty():
        return df
    df = df.sort("timestamp")
    df = df.with_columns(
        [
            pl.col("timestamp").diff().alias("delta"),
        ]
    )
    expected = timedelta(minutes=policy.expected_gap_minutes) if policy.expected_gap_minutes else None
    def classify(row_delta: timedelta | None) -> str:
        if row_delta is None:
            return "none"
        # row_delta is a timedelta
        if expected and row_delta <= expected:
            return "on_schedule"
        return "gap"
    classified = df.with_columns(
        [
            pl.col("delta").map_elements(classify, return_dtype=pl.String).alias("gap_status")
        ]
    ).fill_null("none")
    return classified


def detect_gaps(df: pl.DataFrame, expected_interval: timedelta) -> list[tuple[datetime, datetime]]:
    """Return list of (gap_start, gap_end) where timestamp diff exceeds expected_interval."""
    if df.is_empty() or "timestamp" not in df.columns:
        return []
    df = df.sort("timestamp")
    deltas = df.select(
        [
            pl.col("timestamp"),
            pl.col("timestamp").shift(-1).alias("next_ts"),
            (pl.col("timestamp").shift(-1) - pl.col("timestamp")).alias("delta"),
        ]
    )
    gaps: list[tuple[datetime, datetime]] = []
    for row in deltas.iter_rows(named=True):
        delta = row["delta"]
        if delta is None:
            continue
        if isinstance(delta, timedelta) and delta > expected_interval:
            start = row["timestamp"]
            end = row["next_ts"]
            if isinstance(start, datetime) and isinstance(end, datetime):
                if start.tzinfo is None:
                    start = start.replace(tzinfo=UTC)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=UTC)
                gaps.append((start, end))
    return gaps

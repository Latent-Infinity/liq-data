"""Gap detection and handling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

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
    def classify(row_delta):
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

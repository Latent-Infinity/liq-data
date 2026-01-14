"""Forex-specific normalization helpers."""

from __future__ import annotations

from datetime import UTC, timedelta
from typing import Any

import polars as pl

from liq.data.exceptions import DataQualityError
from liq.data.gaps import detect_gaps


def detect_gap_policy(
    df: pl.DataFrame,
    *,
    expected_minutes: int = 60,
    max_fill_minutes: int = 4320,
) -> dict[str, Any]:
    """Detect gaps and summarize whether they exceed the fill policy."""
    gaps = detect_gaps(df, timedelta(minutes=expected_minutes))
    durations = [int((end - start).total_seconds() / 60) for start, end in gaps]
    max_gap = max(durations) if durations else 0
    oversized = [
        (start, end)
        for (start, end), minutes in zip(gaps, durations)
        if minutes > max_fill_minutes
    ]
    return {
        "expected_minutes": expected_minutes,
        "max_fill_minutes": max_fill_minutes,
        "gap_count": len(gaps),
        "max_gap_minutes": max_gap,
        "oversized_gaps": oversized,
        "within_limit": len(oversized) == 0,
    }


def normalize_hourly(
    df: pl.DataFrame,
    *,
    max_fill_minutes: int = 4320,
) -> pl.DataFrame:
    """Normalize OHLCV to a full 1h UTC grid with synthetic bars."""
    if df.is_empty():
        return df
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise DataQualityError(f"Missing required columns: {sorted(missing)}")

    if df["timestamp"].dtype.time_zone is None:
        raise DataQualityError("Timestamps must be timezone-aware UTC")

    non_monotonic = df.select(
        (pl.col("timestamp") < pl.col("timestamp").shift(1)).any()
    ).item()
    if non_monotonic:
        raise DataQualityError("Timestamps must be monotonic ascending")

    if df.select(pl.col("timestamp").is_duplicated().any()).item():
        raise DataQualityError("Timestamps must be unique")

    gap_summary = detect_gap_policy(df, expected_minutes=60, max_fill_minutes=max_fill_minutes)
    if not gap_summary["within_limit"]:
        raise DataQualityError("Gaps exceed max_fill_minutes policy")

    df = df.sort("timestamp")
    min_ts = df.select(pl.col("timestamp").min()).item()
    max_ts = df.select(pl.col("timestamp").max()).item()
    if min_ts is None or max_ts is None:
        return df

    start = min_ts.replace(minute=0, second=0, microsecond=0)
    end = max_ts.replace(minute=0, second=0, microsecond=0)
    if max_ts > end:
        end = end + timedelta(hours=1)

    full_range = pl.datetime_range(start, end, "1h", time_zone="UTC", eager=True)
    full_df = pl.DataFrame({"timestamp": full_range})

    df = df.with_columns(pl.lit(True).alias("_present"))
    merged = full_df.join(df, on="timestamp", how="left")

    merged = merged.with_columns(
        [
            pl.col("close").forward_fill().alias("_ff_close"),
        ]
    )
    merged = merged.with_columns(
        [
            pl.when(pl.col("_present").is_null())
            .then(pl.col("_ff_close"))
            .otherwise(pl.col("open"))
            .alias("open"),
            pl.when(pl.col("_present").is_null())
            .then(pl.col("_ff_close"))
            .otherwise(pl.col("high"))
            .alias("high"),
            pl.when(pl.col("_present").is_null())
            .then(pl.col("_ff_close"))
            .otherwise(pl.col("low"))
            .alias("low"),
            pl.when(pl.col("_present").is_null())
            .then(pl.col("_ff_close"))
            .otherwise(pl.col("close"))
            .alias("close"),
        ]
    )

    if "volume" in merged.columns:
        merged = merged.with_columns(
            pl.when(pl.col("_present").is_null()).then(0.0).otherwise(pl.col("volume")).alias("volume")
        )
    else:
        merged = merged.with_columns(pl.lit(0.0).alias("volume"))

    merged = merged.with_columns(
        pl.col("_present").is_null().alias("is_synthetic_bar")
    ).drop(["_present", "_ff_close"])

    return merged

"""Aggregation helpers for 1m -> higher timeframe bars."""

from __future__ import annotations

import re
from typing import Literal

import polars as pl

AggregationMethod = Literal["default"]


_TF_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>[mhd])$")


def _timeframe_to_minutes(tf: str) -> int | None:
    """Convert timeframe string to minutes for a set of standard frames."""
    allowed: dict[str, int] = {
        # minutes
        "1m": 1,
        "2m": 2,
        "3m": 3,
        "5m": 5,
        "10m": 10,
        "15m": 15,
        "30m": 30,
        # hours
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "8h": 480,
        "12h": 720,
        # day
        "1d": 1440,
    }
    if tf in allowed:
        return allowed[tf]

    # Permit any whole-minute frame that maps to hours/days to avoid brittle configs
    match = _TF_PATTERN.match(tf)
    if not match:
        return None
    value = int(match.group("value"))
    unit = match.group("unit")
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    return None  # pragma: no cover - unit pattern already restricted


def aggregate_bars(df: pl.DataFrame, timeframe: str, _method: AggregationMethod = "default") -> pl.DataFrame:
    """Aggregate 1m bars to a higher timeframe using provider-mimicking rules.

    Default rule: open=first, high=max, low=min, close=last, volume=sum.

    Supported timeframes include common minutes/hours/day frames (e.g. 1m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d).
    """

    if df.is_empty():
        return df

    # Parse timeframe to minutes
    minutes = _timeframe_to_minutes(timeframe)
    if minutes is None:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. Use standard m/h/d frames like 1m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d"
        )
    if minutes <= 1:
        return df

    # Bucket timestamps aligned to wall-clock boundaries
    df = df.sort("timestamp")
    bucketed = df.with_columns(
        pl.col("timestamp").dt.truncate("1d").alias("_day_start")
    ).with_columns(
        (
            (pl.col("timestamp") - pl.col("_day_start"))
            .dt.total_seconds()
            .floordiv(60)
        ).alias("_minutes_since_day")
    ).with_columns(
        (
            pl.col("_minutes_since_day")
            .floordiv(minutes)
            * minutes
        ).cast(pl.Int64).alias("_bucket_minutes")
    ).with_columns(
        (
            pl.col("_day_start") + pl.duration(minutes=pl.col("_bucket_minutes"))
        ).alias("bucket")
    )

    agg = bucketed.group_by("bucket", maintain_order=True).agg(
        [
            pl.col("bucket").first().alias("timestamp"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]
    )
    return agg.select("timestamp", "open", "high", "low", "close", "volume").sort("timestamp")

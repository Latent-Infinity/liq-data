"""Aggregation helpers for 1m -> higher timeframe bars."""

from __future__ import annotations

from typing import Literal

import polars as pl

AggregationMethod = Literal["default"]


def aggregate_bars(df: pl.DataFrame, timeframe: str, _method: AggregationMethod = "default") -> pl.DataFrame:
    """Aggregate 1m bars to a higher timeframe using provider-mimicking rules.

    Default rule: open=first, high=max, low=min, close=last, volume=sum.
    """

    if df.is_empty():
        return df

    # Only handle minute-based aggregation for now
    if not timeframe.endswith("m"):
        raise ValueError("Only minute-based aggregation supported for now")

    minutes = int(timeframe[:-1])
    if minutes <= 1:
        return df

    # Bucket timestamps
    df = df.sort("timestamp")
    start = df[0, "timestamp"]
    bucketed = df.with_columns(
        (pl.col("timestamp") - start).dt.total_minutes().floordiv(minutes).alias("bucket")
    )

    agg = bucketed.group_by("bucket").agg(
        [
            pl.col("timestamp").first().alias("timestamp"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ]
    )
    return agg.select("timestamp", "open", "high", "low", "close", "volume").sort("timestamp")

"""Bar-level QA utilities."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import polars as pl


@dataclass
class QAResult:
    missing_ratio: float
    zero_volume_ratio: float
    ohlc_inconsistencies: int
    extreme_moves: int
    negative_volume: int
    non_monotonic_ts: int


def run_bar_qa(df: pl.DataFrame) -> QAResult:
    if df.is_empty():
        return QAResult(0.0, 0.0, 0, 0, 0, 0)
    df = df.sort("timestamp")
    total = len(df)
    expected_ts = df.get_column("timestamp")
    missing_ratio = 0.0  # placeholder when we don't know expected calendar
    zero_volume_ratio = (
        df.filter(pl.col("volume") == 0).height / total if total else 0.0
    )
    ohlc_bad = df.filter(
        (pl.col("high") < pl.col("open"))
        | (pl.col("high") < pl.col("close"))
        | (pl.col("low") > pl.col("open"))
        | (pl.col("low") > pl.col("close"))
        | (pl.col("high") < pl.col("low"))
    ).height
    extreme_moves = (
        df.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("ret"))
        .filter(pl.col("ret").abs() > 0.1)
        .height
    )
    negative_volume = df.filter(pl.col("volume") < 0).height
    ts_list = expected_ts.to_list()
    non_monotonic_ts = sum(1 for i in range(1, len(ts_list)) if ts_list[i] < ts_list[i - 1])
    return QAResult(
        missing_ratio=missing_ratio,
        zero_volume_ratio=zero_volume_ratio,
        ohlc_inconsistencies=ohlc_bad,
        extreme_moves=extreme_moves,
        negative_volume=negative_volume,
        non_monotonic_ts=int(non_monotonic_ts),
    )

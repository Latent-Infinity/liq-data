"""Bar-level QA utilities and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from liq.data.exceptions import DataQualityError, SchemaValidationError


@dataclass
class QAResult:
    missing_ratio: float
    zero_volume_ratio: float
    ohlc_inconsistencies: int
    extreme_moves: int
    negative_volume: int
    non_monotonic_ts: int


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]


def run_bar_qa(df: pl.DataFrame) -> QAResult:
    if df.is_empty():
        return QAResult(0.0, 0.0, 0, 0, 0, 0)
    df = df.sort("timestamp")
    total = len(df)
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
    # Count non-monotonic timestamps using vectorized Polars expression
    non_monotonic_ts = df.select(
        (pl.col("timestamp") < pl.col("timestamp").shift(1)).sum()
    ).item()
    return QAResult(
        missing_ratio=missing_ratio,
        zero_volume_ratio=zero_volume_ratio,
        ohlc_inconsistencies=ohlc_bad,
        extreme_moves=extreme_moves,
        negative_volume=negative_volume,
        non_monotonic_ts=int(non_monotonic_ts),
    )


def _require_columns(df: pl.DataFrame, required: set[str]) -> list[str]:
    missing = required - set(df.columns)
    return [f"Missing required columns: {sorted(missing)}"] if missing else []


def _check_ohlc_constraints(df: pl.DataFrame) -> list[str]:
    errs: list[str] = []
    if not df.filter(pl.col("high") < pl.col("low")).is_empty():
        errs.append("Found rows where high < low")
    if not df.filter((pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close"))).is_empty():
        errs.append("Found rows where high < open/close")
    if not df.filter((pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close"))).is_empty():
        errs.append("Found rows where low > open/close")
    if not df.filter(pl.col("volume") < 0).is_empty():
        errs.append("Found rows with negative volume")
    return errs


def _check_spikes(df: pl.DataFrame, threshold: float = 0.2) -> list[str]:
    if "close" not in df.columns:
        return []
    spikes = df.with_columns(((pl.col("close") - pl.col("close").shift(1)).abs() / pl.col("close").shift(1)).alias("pct"))
    if spikes.filter(pl.col("pct") > threshold).is_empty():
        return []
    return ["Price spike warning: move > {:.0%}".format(threshold)]


def _check_stale(df: pl.DataFrame, stale_minutes: int = 60) -> list[str]:
    if "timestamp" not in df.columns or df.is_empty():
        return []
    last = df["timestamp"].max()
    if isinstance(last, datetime) and last.tzinfo:
        delta = datetime.now(tz=UTC) - last
        if delta.total_seconds() / 60 > stale_minutes:
            return [f"Data stale: last timestamp {delta.total_seconds()/60:.1f} minutes ago"]
    return []


def validate_ohlc(df: pl.DataFrame) -> ValidationResult:
    """Validate OHLC data against PRD rules."""
    errors: list[str] = []
    warnings: list[str] = []

    if df.is_empty():
        warnings.append("DataFrame is empty")
        return ValidationResult(is_valid=True, errors=errors, warnings=warnings, stats={})

    errors.extend(_require_columns(df, {"timestamp", "open", "high", "low", "close", "volume"}))
    if errors:
        raise SchemaValidationError(errors[0])

    if df["timestamp"].dtype.time_zone is None:
        errors.append("Timestamps must be timezone-aware UTC")

    errors.extend(_check_ohlc_constraints(df))
    warnings.extend(_check_spikes(df))
    warnings.extend(_check_stale(df))

    stats = {
        "row_count": df.height,
        "start_ts": df["timestamp"].min(),
        "end_ts": df["timestamp"].max(),
    }

    if errors:
        raise DataQualityError("; ".join(errors))

    return ValidationResult(is_valid=True, errors=errors, warnings=warnings, stats=stats)

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from liq.data.exceptions import DataQualityError
from liq.data.forex import detect_gap_policy, normalize_hourly


def _sample_df(timestamps: list[datetime]) -> pl.DataFrame:
    base = [1.1000 + i * 0.0001 for i in range(len(timestamps))]
    return pl.DataFrame({
        "timestamp": timestamps,
        "open": base,
        "high": [v + 0.0002 for v in base],
        "low": [v - 0.0002 for v in base],
        "close": [v + 0.00005 for v in base],
        "volume": [1000 + i for i in range(len(timestamps))],
    })


def test_normalize_hourly_fills_gaps() -> None:
    ts0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    ts2 = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
    df = _sample_df([ts0, ts2])
    out = normalize_hourly(df, max_fill_minutes=180)
    assert out.height == 3
    assert out.filter(pl.col("is_synthetic_bar")).height == 1
    synth = out.filter(pl.col("is_synthetic_bar")).row(0, named=True)
    assert synth["open"] == synth["close"]
    assert synth["high"] == synth["close"]
    assert synth["low"] == synth["close"]
    assert synth["volume"] == 0.0


def test_normalize_hourly_rejects_large_gap() -> None:
    ts0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    ts_far = ts0 + timedelta(hours=100)
    df = _sample_df([ts0, ts_far])
    with pytest.raises(DataQualityError):
        normalize_hourly(df, max_fill_minutes=72 * 60)


def test_normalize_hourly_requires_utc() -> None:
    ts0 = datetime(2024, 1, 1, 0, 0)
    ts1 = datetime(2024, 1, 1, 1, 0)
    df = _sample_df([ts0, ts1])
    with pytest.raises(DataQualityError):
        normalize_hourly(df)


def test_detect_gap_policy_summary() -> None:
    ts0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    ts2 = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
    df = _sample_df([ts0, ts2])
    summary = detect_gap_policy(df, expected_minutes=60, max_fill_minutes=180)
    assert summary["gap_count"] == 1
    assert summary["within_limit"] is True

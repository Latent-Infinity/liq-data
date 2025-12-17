"""Tests for aggregation helpers."""

from datetime import datetime, UTC

import polars as pl
import pytest

from polars.testing import assert_frame_equal

from liq.data.aggregation import aggregate_bars


def test_aggregate_empty_returns_empty() -> None:
    """Empty frames should pass through unchanged."""
    df = pl.DataFrame({"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
    result = aggregate_bars(df, "5m")
    assert result.is_empty()


def test_aggregate_only_minutes_supported() -> None:
    """Unsupported formats should raise."""
    df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
        "open": [1],
        "high": [1],
        "low": [1],
        "close": [1],
        "volume": [1],
    })
    with pytest.raises(ValueError):
        aggregate_bars(df, "90s")


def test_aggregate_buckets_prices() -> None:
    """Simple aggregation should roll up by bucket."""
    df = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
        ],
        "open": [1.0, 1.1, 1.2],
        "high": [1.1, 1.2, 1.3],
        "low": [0.9, 1.0, 1.1],
        "close": [1.05, 1.15, 1.25],
        "volume": [10, 20, 30],
    })

    result = aggregate_bars(df, "2m")
    assert result.height == 2
    assert result["volume"].to_list() == [30, 30]


def test_aggregate_pass_through_for_one_minute() -> None:
    """Requesting 1m returns input unchanged."""
    df = pl.DataFrame({
        "timestamp": [datetime(2024, 1, 1, 0, 0, tzinfo=UTC)],
        "open": [1.0],
        "high": [1.0],
        "low": [1.0],
        "close": [1.0],
        "volume": [10],
    })

    result = aggregate_bars(df, "1m")
    assert_frame_equal(result, df)


def test_dynamic_hour_frame_supported_via_pattern() -> None:
    """Frames not enumerated but parseable (e.g. 7h) should work."""
    df = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
        ],
        "open": [1.0, 1.1, 1.2],
        "high": [1.1, 1.2, 1.3],
        "low": [0.9, 1.0, 1.1],
        "close": [1.05, 1.15, 1.25],
        "volume": [10, 20, 30],
    })

    result = aggregate_bars(df, "7h")
    assert result.height == 1


@pytest.mark.parametrize("timeframe, expected_rows", [
    ("5m", 1),
    ("15m", 1),
    ("30m", 1),
    ("1h", 1),
    ("2h", 1),
    ("4h", 1),
    ("8h", 1),
    ("12h", 1),
    ("1d", 1),
    ("7m", 1),
    ("2d", 1),
])
def test_supported_timeframes_cover_standard_set(timeframe: str, expected_rows: int) -> None:
    df = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 2, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 3, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 4, tzinfo=UTC),
        ],
        "open": [1.0, 1.1, 1.2, 1.3, 1.4],
        "high": [1.1, 1.2, 1.3, 1.4, 1.5],
        "low": [0.9, 1.0, 1.1, 1.2, 1.3],
        "close": [1.05, 1.15, 1.25, 1.35, 1.45],
        "volume": [10, 20, 30, 40, 50],
    })

    result = aggregate_bars(df, timeframe)
    assert result.height == expected_rows
    # ensure no data loss
    assert result["volume"].sum() == sum(df["volume"])

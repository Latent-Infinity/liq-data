"""Tests for aggregation helpers."""

from datetime import datetime, UTC

import polars as pl
import pytest

from liq.data.aggregation import aggregate_bars


def test_aggregate_empty_returns_empty() -> None:
    """Empty frames should pass through unchanged."""
    df = pl.DataFrame({"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
    result = aggregate_bars(df, "5m")
    assert result.is_empty()


def test_aggregate_only_minutes_supported() -> None:
    """Non-minute timeframe should raise for now."""
    df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})
    with pytest.raises(ValueError):
        aggregate_bars(df, "1h")


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

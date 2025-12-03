import polars as pl
from datetime import datetime, timezone

from liq.data.aggregation import aggregate_bars


def test_aggregate_default_rules() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 4, tzinfo=timezone.utc),
            ],
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [10, 10, 10, 10, 10],
        }
    )
    agg = aggregate_bars(df, "5m")
    assert agg.height == 1
    row = agg.row(0, named=True)
    assert row["open"] == 1
    assert row["high"] == 6
    assert row["low"] == 0
    assert row["close"] == 5.5
    assert row["volume"] == 50

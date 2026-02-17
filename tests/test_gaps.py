from datetime import UTC, datetime

import polars as pl

from liq.data.gaps import GapPolicy, classify_gaps


def test_gap_classification_marks_gaps() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 10, tzinfo=UTC),
            ],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [1, 1, 1],
        }
    )
    policy = GapPolicy(expected_gap_minutes=1, max_forward_fill_minutes=5)
    out = classify_gaps(df, policy)
    statuses = out.get_column("gap_status").to_list()
    assert statuses[0] == "none"
    assert statuses[1] == "on_schedule"
    assert statuses[2] == "gap"

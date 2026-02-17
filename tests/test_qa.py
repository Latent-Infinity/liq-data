from datetime import UTC, datetime

import polars as pl

from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE
from liq.data.qa import run_bar_qa


def test_qa_detects_issues() -> None:
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            ],
            "open": [100.0, 110.0],
            "high": [99.0, 111.0],  # first row high < open -> inconsistency
            "low": [95.0, 109.0],
            "close": [98.0, 112.0],
            "volume": [0.0, -5.0],
        }
    )
    qa = run_bar_qa(df)
    assert qa.zero_volume_ratio == 0.5
    assert qa.negative_volume == 1
    assert qa.ohlc_inconsistencies == 2
    assert qa.extreme_moves == 1  # 12% move
    assert qa.non_monotonic_ts == 0


def test_qa_empty_df_safe() -> None:
    df = pl.DataFrame(
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "open": PRICE_DTYPE,
            "high": PRICE_DTYPE,
            "low": PRICE_DTYPE,
            "close": PRICE_DTYPE,
            "volume": VOLUME_DTYPE,
        }
    )
    qa = run_bar_qa(df)
    assert qa.missing_ratio == 0.0
    assert qa.zero_volume_ratio == 0.0

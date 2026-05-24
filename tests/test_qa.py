from datetime import UTC, datetime
from decimal import Decimal

import polars as pl

from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE
from liq.data.qa import run_bar_qa, validate_ohlc


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


def _zero_crossing_df() -> pl.DataFrame:
    """Decimal-typed OHLC where a close is exactly 0 followed by a nonzero.

    Mirrors FRED macro series (e.g. T10Y2Y yield-curve spread) that legitimately
    cross zero. Polars Decimal division-by-zero *raises* (unlike float), so QA
    return calcs (close / close.shift(1)) must guard the denominator.
    """
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ],
            "open": [Decimal("0.05"), Decimal("0.00"), Decimal("0.03")],
            "high": [Decimal("0.05"), Decimal("0.00"), Decimal("0.03")],
            "low": [Decimal("0.05"), Decimal("0.00"), Decimal("0.03")],
            "close": [Decimal("0.05"), Decimal("0.00"), Decimal("0.03")],
            "volume": [Decimal("0"), Decimal("0"), Decimal("0")],
        },
        schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "open": PRICE_DTYPE,
            "high": PRICE_DTYPE,
            "low": PRICE_DTYPE,
            "close": PRICE_DTYPE,
            "volume": VOLUME_DTYPE,
        },
    )


def test_run_bar_qa_handles_zero_prior_close_without_crashing() -> None:
    qa = run_bar_qa(_zero_crossing_df())  # must not raise Decimal div-by-zero
    assert qa.ohlc_inconsistencies == 0
    # close 0.03 vs prior close 0.00 -> return undefined -> not an extreme move
    assert isinstance(qa.extreme_moves, int)


def test_validate_ohlc_handles_zero_prior_close_without_crashing() -> None:
    result = validate_ohlc(_zero_crossing_df())  # _check_spikes divides by prev
    assert result.is_valid

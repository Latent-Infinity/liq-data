"""Tests for temporal correctness and data integrity.

Following TDD: Tests verify timestamp handling and data quality checks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE, BaseProvider
from liq.data.qa import QAResult, run_bar_qa
from liq.data.gaps import GapPolicy, classify_gaps


class TestTimestampOrdering:
    """Tests for timestamp ordering validation.

    Note: run_bar_qa sorts data before checking non_monotonic_ts,
    so it checks for issues after sorting (which should be 0 for valid data).
    The purpose is to verify the sorted data is monotonic.
    """

    def test_qa_sorted_data_is_monotonic(self) -> None:
        """QA should show 0 non-monotonic after sorting."""
        # Even if input is out of order, after sorting it should be monotonic
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0],
        })

        qa = run_bar_qa(df)

        # After sorting, data should be monotonic
        assert qa.non_monotonic_ts == 0

    def test_qa_accepts_monotonic_timestamps(self) -> None:
        """QA should pass for correctly ordered timestamps."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0],
        })

        qa = run_bar_qa(df)

        assert qa.non_monotonic_ts == 0

    def test_qa_handles_duplicate_timestamps(self) -> None:
        """QA should not flag duplicate timestamps as non-monotonic."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),  # Duplicate
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
            ],
            "open": [100.0, 100.0, 102.0],
            "high": [105.0, 105.0, 107.0],
            "low": [95.0, 95.0, 97.0],
            "close": [102.0, 102.0, 104.0],
            "volume": [1000.0, 1000.0, 1200.0],
        })

        qa = run_bar_qa(df)

        # Duplicates are not "less than" previous, so should be 0
        assert qa.non_monotonic_ts == 0


class TestTimezoneEnforcement:
    """Tests for UTC timezone enforcement in providers."""

    def test_base_provider_enforces_utc(self) -> None:
        """BaseProvider.bars_to_dataframe should enforce UTC timezone."""
        bars = [
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0),  # Naive datetime
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ]

        df = BaseProvider.bars_to_dataframe(bars)

        # Should be cast to UTC
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")

    def test_base_provider_preserves_utc(self) -> None:
        """BaseProvider should preserve existing UTC timezone."""
        bars = [
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        ]

        df = BaseProvider.bars_to_dataframe(bars)

        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")

    def test_empty_dataframe_has_utc_schema(self) -> None:
        """Empty DataFrame should have UTC timezone in schema."""
        df = BaseProvider.bars_to_dataframe([])

        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        assert df.is_empty()


class TestOHLCIntegrity:
    """Tests for OHLC data integrity checks."""

    def test_qa_detects_high_less_than_open(self) -> None:
        """QA should detect when high < open."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [99.0],  # Invalid: high < open
            "low": [95.0],
            "close": [98.0],
            "volume": [1000.0],
        })

        qa = run_bar_qa(df)

        assert qa.ohlc_inconsistencies == 1

    def test_qa_detects_high_less_than_close(self) -> None:
        """QA should detect when high < close."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [101.0],
            "low": [95.0],
            "close": [102.0],  # Invalid: close > high
            "volume": [1000.0],
        })

        qa = run_bar_qa(df)

        assert qa.ohlc_inconsistencies == 1

    def test_qa_detects_low_greater_than_open(self) -> None:
        """QA should detect when low > open."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [105.0],
            "low": [101.0],  # Invalid: low > open
            "close": [102.0],
            "volume": [1000.0],
        })

        qa = run_bar_qa(df)

        assert qa.ohlc_inconsistencies == 1

    def test_qa_detects_low_greater_than_close(self) -> None:
        """QA should detect when low > close."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [98.0],  # Invalid: close < low
            "volume": [1000.0],
        })

        qa = run_bar_qa(df)

        assert qa.ohlc_inconsistencies == 1

    def test_qa_detects_high_less_than_low(self) -> None:
        """QA should detect when high < low."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [95.0],  # Invalid: high < low
            "low": [99.0],
            "close": [97.0],
            "volume": [1000.0],
        })

        qa = run_bar_qa(df)

        assert qa.ohlc_inconsistencies >= 1

    def test_qa_passes_valid_ohlc(self) -> None:
        """QA should pass for valid OHLC data."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [1000.0],
        })

        qa = run_bar_qa(df)

        assert qa.ohlc_inconsistencies == 0


class TestGapClassification:
    """Tests for gap detection in time series."""

    def test_gaps_detects_missing_bar(self) -> None:
        """Should detect gap when bar is missing."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                # 0:02 is missing
                datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0],
        })

        policy = GapPolicy(expected_gap_minutes=1)
        result = classify_gaps(df, policy)

        # Third row should be classified as gap (2 min delta > 1 min expected)
        gap_statuses = result["gap_status"].to_list()
        assert gap_statuses[-1] == "gap"

    def test_gaps_accepts_on_schedule(self) -> None:
        """Should accept bars that arrive on schedule."""
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0],
        })

        policy = GapPolicy(expected_gap_minutes=1)
        result = classify_gaps(df, policy)

        # All bars after first should be on_schedule
        gap_statuses = result["gap_status"].to_list()
        assert gap_statuses[1] == "on_schedule"
        assert gap_statuses[2] == "on_schedule"

    def test_gaps_empty_dataframe(self) -> None:
        """Should handle empty DataFrame."""
        df = pl.DataFrame(schema={
            "timestamp": pl.Datetime("us", "UTC"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        })

        policy = GapPolicy(expected_gap_minutes=1)
        result = classify_gaps(df, policy)

        assert result.is_empty()


class TestDecimalPrecision:
    """Tests for financial precision in data handling."""

    def test_price_decimal_precision(self) -> None:
        """Price columns should use Decimal for precision."""
        bars = [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "open": 0.00000001,  # 8 decimal places (crypto precision)
                "high": 0.00000002,
                "low": 0.00000001,
                "close": 0.00000002,
                "volume": 1000000.00,
            }
        ]

        df = BaseProvider.bars_to_dataframe(bars)

        # Should preserve 8 decimal places
        assert df["open"].dtype == PRICE_DTYPE
        assert df["high"].dtype == PRICE_DTYPE
        assert df["low"].dtype == PRICE_DTYPE
        assert df["close"].dtype == PRICE_DTYPE

    def test_volume_decimal_precision(self) -> None:
        """Volume should use Decimal for precision."""
        bars = [
            {
                "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1234567890.12,  # Large volume with decimals
            }
        ]

        df = BaseProvider.bars_to_dataframe(bars)

        assert df["volume"].dtype == VOLUME_DTYPE

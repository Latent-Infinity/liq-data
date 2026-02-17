"""Tests for liq.data.updater module."""

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from liq.data.exceptions import ProviderError
from liq.data.updater import IncrementalUpdater


@pytest.fixture
def mock_fetcher() -> MagicMock:
    """Create a mock DataFetcher."""
    fetcher = MagicMock()
    fetcher.fetch_and_store.return_value = 100
    return fetcher


@pytest.fixture
def mock_store() -> MagicMock:
    """Create a mock store."""
    store = MagicMock()
    return store


class TestIncrementalUpdaterCreation:
    """Tests for IncrementalUpdater instantiation."""

    def test_create_updater(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test IncrementalUpdater creation."""
        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        assert updater.fetcher is mock_fetcher
        assert updater.store is mock_store

    def test_default_asset_class(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test default asset_class is forex."""
        updater = IncrementalUpdater(fetcher=mock_fetcher, store=mock_store)

        assert updater._asset_class == "forex"


class TestIncrementalUpdaterDetectGaps:
    """Tests for IncrementalUpdater.detect_gaps method."""

    def test_detect_gaps_no_existing_data(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test gap detection with no existing data."""
        mock_store.get_date_range.return_value = None

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        gaps = updater.detect_gaps(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert len(gaps) == 1
        assert gaps[0] == (date(2024, 1, 1), date(2024, 1, 10))

    def test_detect_gaps_complete_coverage(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test gap detection with complete data coverage."""
        mock_store.get_date_range.return_value = (date(2024, 1, 1), date(2024, 1, 10))

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        gaps = updater.detect_gaps(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert len(gaps) == 0

    def test_detect_gaps_before_existing(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test gap detection before existing data."""
        mock_store.get_date_range.return_value = (date(2024, 1, 5), date(2024, 1, 10))

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        gaps = updater.detect_gaps(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert len(gaps) == 1
        assert gaps[0][0] == date(2024, 1, 1)
        assert gaps[0][1] == date(2024, 1, 4)

    def test_detect_gaps_after_existing(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test gap detection after existing data."""
        mock_store.get_date_range.return_value = (date(2024, 1, 1), date(2024, 1, 5))

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        gaps = updater.detect_gaps(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert len(gaps) == 1
        assert gaps[0][0] == date(2024, 1, 6)
        assert gaps[0][1] == date(2024, 1, 10)

    def test_detect_gaps_both_sides(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test gap detection on both sides of existing data."""
        mock_store.get_date_range.return_value = (date(2024, 1, 5), date(2024, 1, 8))

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        gaps = updater.detect_gaps(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 15),
            timeframe="1d",
        )

        assert len(gaps) == 2
        # Gap before
        assert gaps[0][0] == date(2024, 1, 1)
        assert gaps[0][1] == date(2024, 1, 4)
        # Gap after
        assert gaps[1][0] == date(2024, 1, 9)
        assert gaps[1][1] == date(2024, 1, 15)


class TestIncrementalUpdaterUpdate:
    """Tests for IncrementalUpdater.update method."""

    def test_update_no_gaps(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update with no gaps returns zero."""
        mock_store.get_date_range.return_value = (date(2024, 1, 1), date(2024, 1, 10))

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        result = updater.update(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert result.success is True
        assert result.gaps_filled == 0
        assert result.total_rows == 0
        mock_fetcher.fetch_and_store.assert_not_called()

    def test_update_fills_gaps(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update fills detected gaps."""
        mock_store.get_date_range.return_value = None
        mock_fetcher.fetch_and_store.return_value = 100

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        result = updater.update(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert result.success is True
        assert result.gaps_filled == 1
        assert result.total_rows == 100
        mock_fetcher.fetch_and_store.assert_called_once()

    def test_update_fills_multiple_gaps(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update fills multiple gaps."""
        mock_store.get_date_range.return_value = (date(2024, 1, 5), date(2024, 1, 8))
        mock_fetcher.fetch_and_store.return_value = 50

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        result = updater.update(
            "EUR_USD",
            date(2024, 1, 1),
            date(2024, 1, 15),
            timeframe="1d",
        )

        assert result.success is True
        assert result.gaps_filled == 2
        assert result.total_rows == 100  # 2 gaps * 50 rows each


class TestIncrementalUpdaterUpdateMultiple:
    """Tests for IncrementalUpdater.update_multiple method."""

    def test_update_multiple_success(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update multiple symbols."""
        mock_store.get_date_range.return_value = None
        mock_fetcher.fetch_and_store.return_value = 100

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        results = updater.update_multiple(
            ["EUR_USD", "GBP_USD"],
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert results.total == 2
        assert results.succeeded == 2
        assert results.failed == 0
        eur_result = next(r for r in results.results if r.symbol == "EUR_USD")
        gbp_result = next(r for r in results.results if r.symbol == "GBP_USD")
        assert eur_result.success is True
        assert eur_result.gaps_filled == 1
        assert gbp_result.success is True

    def test_update_multiple_partial_failure(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test partial failure in update multiple."""
        mock_store.get_date_range.return_value = None

        call_count = {"n": 0}

        def fetch_side_effect(*args, **kwargs) -> int:
            call_count["n"] += 1
            if call_count["n"] == 2:  # Second call fails
                raise ProviderError("Failed")
            return 100

        mock_fetcher.fetch_and_store.side_effect = fetch_side_effect

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        results = updater.update_multiple(
            ["EUR_USD", "BAD_SYMBOL", "GBP_USD"],
            date(2024, 1, 1),
            date(2024, 1, 10),
            timeframe="1d",
        )

        assert results.total == 3
        assert results.succeeded == 2
        assert results.failed == 1
        eur_result = next(r for r in results.results if r.symbol == "EUR_USD")
        bad_result = next(r for r in results.results if r.symbol == "BAD_SYMBOL")
        gbp_result = next(r for r in results.results if r.symbol == "GBP_USD")
        assert eur_result.success is True
        assert bad_result.success is False
        assert bad_result.error is not None
        assert gbp_result.success is True


class TestIncrementalUpdaterUpdateToNow:
    """Tests for IncrementalUpdater.update_to_now method."""

    def test_update_to_now_no_existing_data(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update_to_now with no existing data uses lookback."""
        mock_store.get_date_range.return_value = None
        mock_fetcher.fetch_and_store.return_value = 100

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        result = updater.update_to_now("EUR_USD", lookback_days=7, timeframe="1d")

        assert result.success is True
        assert result.gaps_filled == 1
        mock_fetcher.fetch_and_store.assert_called_once()

        # Verify start date is approximately 7 days ago
        call_args = mock_fetcher.fetch_and_store.call_args
        start_date = call_args[0][1]
        expected_start = date.today() - timedelta(days=7)
        assert start_date == expected_start

    def test_update_to_now_with_existing_data(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update_to_now with existing data starts from last date."""
        # Existing data ends 5 days ago
        existing_end = date.today() - timedelta(days=5)
        mock_store.get_date_range.return_value = (
            date.today() - timedelta(days=30),
            existing_end,
        )
        mock_fetcher.fetch_and_store.return_value = 100

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        result = updater.update_to_now("EUR_USD", timeframe="1d")

        assert result.success is True
        mock_fetcher.fetch_and_store.assert_called_once()

        # Verify start date is day after existing end
        call_args = mock_fetcher.fetch_and_store.call_args
        start_date = call_args[0][1]
        assert start_date == existing_end + timedelta(days=1)

    def test_update_to_now_already_current(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update_to_now returns early if data is current."""
        mock_store.get_date_range.return_value = (
            date.today() - timedelta(days=30),
            date.today(),
        )

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        result = updater.update_to_now("EUR_USD", timeframe="1d")

        assert result.success is True
        assert result.gaps_filled == 0
        assert result.total_rows == 0
        mock_fetcher.fetch_and_store.assert_not_called()

    def test_update_to_now_error_handling(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test update_to_now handles errors gracefully."""
        mock_store.get_date_range.return_value = None
        mock_fetcher.fetch_and_store.side_effect = ProviderError("API error")

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        result = updater.update_to_now("EUR_USD", timeframe="1d")

        assert result.success is False
        assert result.error is not None


class TestIncrementalUpdaterTimeframeConversion:
    """Tests for timeframe to timedelta conversion."""

    def test_supported_timeframes(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test all supported timeframes are valid."""
        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        supported = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        for tf in supported:
            delta = updater._timeframe_to_timedelta(tf)
            assert delta is not None

    def test_unsupported_timeframe(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test unsupported timeframe raises ValueError."""
        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            updater._timeframe_to_timedelta("invalid")


class TestDetectInternalGaps:
    """Tests for detecting gaps within existing data."""

    def test_detect_internal_gaps_no_gaps(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test detect_internal_gaps with contiguous data."""
        from datetime import datetime

        import polars as pl

        # Create DataFrame with no gaps (1-minute data)
        timestamps = [
            datetime(2024, 1, 15, 10, 0),
            datetime(2024, 1, 15, 10, 1),
            datetime(2024, 1, 15, 10, 2),
            datetime(2024, 1, 15, 10, 3),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [100.0, 200.0, 300.0, 400.0],
        })

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = updater.detect_internal_gaps(df, timeframe="1m")

        assert len(gaps) == 0

    def test_detect_internal_gaps_single_gap(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test detect_internal_gaps with a single gap."""
        from datetime import datetime

        import polars as pl

        # Create DataFrame with a gap (missing 10:2 and 10:3)
        timestamps = [
            datetime(2024, 1, 15, 10, 0),
            datetime(2024, 1, 15, 10, 1),
            # Gap: 10:2 and 10:3 missing
            datetime(2024, 1, 15, 10, 4),
            datetime(2024, 1, 15, 10, 5),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [100.0, 200.0, 300.0, 400.0],
        })

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = updater.detect_internal_gaps(df, timeframe="1m")

        assert len(gaps) == 1
        # Gap should span from after 10:1 to before 10:4
        gap_start, gap_end = gaps[0]
        assert gap_start == datetime(2024, 1, 15, 10, 2)
        assert gap_end == datetime(2024, 1, 15, 10, 3)

    def test_detect_internal_gaps_multiple_gaps(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test detect_internal_gaps with multiple gaps."""
        from datetime import datetime

        import polars as pl

        timestamps = [
            datetime(2024, 1, 15, 10, 0),
            # Gap 1: 10:1-10:2 missing
            datetime(2024, 1, 15, 10, 3),
            datetime(2024, 1, 15, 10, 4),
            # Gap 2: 10:5-10:7 missing
            datetime(2024, 1, 15, 10, 8),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [100.0, 200.0, 300.0, 400.0],
        })

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = updater.detect_internal_gaps(df, timeframe="1m")

        assert len(gaps) == 2

    def test_detect_internal_gaps_hourly_timeframe(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test detect_internal_gaps with hourly data."""
        from datetime import datetime

        import polars as pl

        timestamps = [
            datetime(2024, 1, 15, 10, 0),
            datetime(2024, 1, 15, 11, 0),
            # Gap: 12:00 missing
            datetime(2024, 1, 15, 13, 0),
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [100.0, 200.0, 300.0],
        })

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = updater.detect_internal_gaps(df, timeframe="1h")

        assert len(gaps) == 1
        gap_start, gap_end = gaps[0]
        assert gap_start == datetime(2024, 1, 15, 12, 0)
        assert gap_end == datetime(2024, 1, 15, 12, 0)

    def test_detect_internal_gaps_ignores_weekend_forex(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test detect_internal_gaps ignores weekend gaps for forex."""
        from datetime import datetime

        import polars as pl

        # Friday to Monday data (weekend gap is expected for forex)
        timestamps = [
            datetime(2024, 1, 12, 21, 0),  # Friday 9 PM
            # Weekend gap is expected
            datetime(2024, 1, 15, 0, 0),  # Monday midnight
        ]
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 200.0],
        })

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
            asset_class="forex",
        )

        gaps = updater.detect_internal_gaps(df, timeframe="1h", skip_weekends=True)

        # Weekend gap should be ignored for forex
        assert len(gaps) == 0

    def test_detect_internal_gaps_empty_dataframe(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test detect_internal_gaps with empty DataFrame."""
        import polars as pl

        df = pl.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }).with_columns(pl.col("timestamp").cast(pl.Datetime))

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = updater.detect_internal_gaps(df, timeframe="1m")

        assert len(gaps) == 0

    def test_detect_internal_gaps_single_bar(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test detect_internal_gaps with single bar."""
        from datetime import datetime

        import polars as pl

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 10, 0)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = updater.detect_internal_gaps(df, timeframe="1m")

        assert len(gaps) == 0


class TestBackfillGaps:
    """Tests for backfilling detected gaps."""

    def test_backfill_single_gap(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test backfilling a single gap."""
        from datetime import datetime

        mock_fetcher.fetch_and_store.return_value = 2

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = [(datetime(2024, 1, 15, 10, 2), datetime(2024, 1, 15, 10, 3))]

        result = updater.backfill_gaps("EUR_USD", gaps, timeframe="1m")

        assert result.success is True
        assert result.gaps_filled == 1
        assert result.total_rows == 2
        mock_fetcher.fetch_and_store.assert_called_once()

    def test_backfill_multiple_gaps(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test backfilling multiple gaps."""
        from datetime import datetime

        mock_fetcher.fetch_and_store.return_value = 3

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = [
            (datetime(2024, 1, 15, 10, 2), datetime(2024, 1, 15, 10, 3)),
            (datetime(2024, 1, 15, 11, 5), datetime(2024, 1, 15, 11, 7)),
        ]

        result = updater.backfill_gaps("EUR_USD", gaps, timeframe="1m")

        assert result.success is True
        assert result.gaps_filled == 2
        assert result.total_rows == 6  # 3 rows per gap
        assert mock_fetcher.fetch_and_store.call_count == 2

    def test_backfill_no_gaps(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test backfill with no gaps returns early."""
        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        result = updater.backfill_gaps("EUR_USD", [], timeframe="1m")

        assert result.success is True
        assert result.gaps_filled == 0
        assert result.total_rows == 0
        mock_fetcher.fetch_and_store.assert_not_called()

    def test_backfill_respects_provider_limits(
        self, mock_fetcher: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test backfill handles provider errors gracefully."""
        from datetime import datetime

        mock_fetcher.fetch_and_store.side_effect = ProviderError("Rate limit")

        updater = IncrementalUpdater(
            fetcher=mock_fetcher,
            store=mock_store,
        )

        gaps = [(datetime(2024, 1, 15, 10, 2), datetime(2024, 1, 15, 10, 3))]

        result = updater.backfill_gaps("EUR_USD", gaps, timeframe="1m")

        assert result.success is False
        assert result.error is not None

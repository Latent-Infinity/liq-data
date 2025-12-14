"""Incremental data updater for gap detection and backfilling.

This module provides intelligent gap detection in existing data and
backfills only the missing ranges, avoiding redundant fetching.

Design Principles:
- SRP: Focused solely on gap detection and incremental updates
- DRY: Centralized gap detection logic
- KISS: Simple interface for updating data incrementally

Example:
    from liq.data.fetcher import DataFetcher
    from liq.data.updater import IncrementalUpdater
    from liq.data.providers.oanda import OandaProvider
    from liq.store import ParquetStore

    provider = OandaProvider(api_key="...", account_id="...", environment="practice")
    store = ParquetStore(data_root="./data")
    fetcher = DataFetcher(provider=provider, store=store, asset_class="forex")

    updater = IncrementalUpdater(fetcher=fetcher, store=store, asset_class="forex")

    # Detect and fill gaps
    result = updater.update(
        symbol="EUR_USD",
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
        timeframe="1h"
    )

    # Update to current date
    result = updater.update_to_now("EUR_USD")

    # Detect and backfill internal gaps using data from liq-store
    storage_key = "forex/EUR_USD"
    df = store.read(storage_key)
    gaps = updater.detect_internal_gaps(df, timeframe="1h")
    result = updater.backfill_gaps("EUR_USD", gaps, timeframe="1h")
"""

import logging
from datetime import date, datetime, timedelta

import polars as pl

from liq.core import BatchResult, UpdateResult
from liq.data.exceptions import DataError
from liq.data.fetcher import DataFetcher
from liq.store.protocols import TimeSeriesStore

logger = logging.getLogger(__name__)

# Weekend gap detection thresholds (in hours)
MIN_WEEKEND_GAP_HOURS = 48  # Minimum expected weekend duration
MAX_WEEKEND_GAP_HOURS = 72  # Maximum expected weekend duration


class IncrementalUpdater:
    """Detects gaps in existing data and backfills missing ranges.

    This class analyzes existing data coverage and intelligently fetches
    only the missing data ranges, avoiding redundant API calls and storage
    operations.
    """

    # Mapping of timeframe strings to timedelta increments
    _TIMEFRAME_TO_DELTA: dict[str, timedelta] = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    def __init__(
        self,
        fetcher: DataFetcher,
        store: TimeSeriesStore,
        asset_class: str = "forex",
    ) -> None:
        """Initialize IncrementalUpdater with fetcher and storage.

        Args:
            fetcher: DataFetcher instance for fetching data
            store: Storage backend instance for querying existing data
            asset_class: Asset class for storage key generation (default: "forex")
        """
        self._fetcher = fetcher
        self._store = store
        self._asset_class = asset_class

    @property
    def fetcher(self) -> DataFetcher:
        """Get the data fetcher."""
        return self._fetcher

    @property
    def store(self) -> TimeSeriesStore:
        """Get the storage backend."""
        return self._store

    def detect_gaps(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> list[tuple[date, date]]:
        """Detect gaps in existing data for the given range.

        Compares requested range with existing data coverage and returns
        list of missing date ranges.

        Args:
            symbol: Canonical symbol (e.g., "EUR_USD")
            start: Start of requested range (inclusive)
            end: End of requested range (inclusive)
            timeframe: Candle timeframe that determines expected spacing

        Returns:
            List of (start, end) tuples representing gaps
            Empty list if data is complete
            [(start, end)] if no data exists
        """
        storage_key = f"{self._asset_class}/{symbol}"
        interval = self._timeframe_to_timedelta(timeframe)

        # Check if data exists
        date_range = self._store.get_date_range(storage_key)

        if date_range is None:
            # No existing data, return full range as gap
            logger.info(
                "no existing data symbol=%s asset_class=%s",
                symbol,
                self._asset_class,
            )
            return [(start, end)]

        existing_start, existing_end = date_range

        logger.info(
            "existing data found symbol=%s existing_start=%s existing_end=%s",
            symbol,
            existing_start.isoformat(),
            existing_end.isoformat(),
        )

        gaps: list[tuple[date, date]] = []

        # Gap before existing data
        if start < existing_start:
            gap_end = min(existing_start - timedelta(days=1), end)
            if gap_end >= start:
                gaps.append((start, gap_end))

        # Gap after existing data (accounting for interval)
        interval_days = max(1, interval.days)
        if end > existing_end:
            gap_start = max(existing_end + timedelta(days=interval_days), start)
            if gap_start <= end:
                gaps.append((gap_start, end))

        if gaps:
            logger.info(
                "gaps detected symbol=%s gap_count=%d",
                symbol,
                len(gaps),
            )
        else:
            logger.info("no gaps detected symbol=%s", symbol)

        return gaps

    def update(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> UpdateResult:
        """Update data by detecting and filling gaps.

        Only fetches missing data ranges, avoiding redundant API calls.

        Args:
            symbol: Canonical symbol (e.g., "EUR_USD")
            start: Start of requested range (inclusive)
            end: End of requested range (inclusive)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            UpdateResult with gaps_filled and total_rows on success.
        """
        logger.info(
            "starting incremental update symbol=%s start=%s end=%s",
            symbol,
            start.isoformat(),
            end.isoformat(),
        )

        # Detect gaps
        gaps = self.detect_gaps(symbol, start, end, timeframe=timeframe)

        if not gaps:
            logger.info("no gaps to fill symbol=%s", symbol)
            return UpdateResult(
                symbol=symbol, success=True, gaps_filled=0, total_rows=0
            )

        # Fill each gap
        total_rows = 0
        for gap_start, gap_end in gaps:
            logger.info(
                "filling gap symbol=%s gap_start=%s gap_end=%s",
                symbol,
                gap_start.isoformat(),
                gap_end.isoformat(),
            )

            rows = self._fetcher.fetch_and_store(symbol, gap_start, gap_end, timeframe)
            total_rows += rows

        logger.info(
            "incremental update complete symbol=%s gaps_filled=%d total_rows=%d",
            symbol,
            len(gaps),
            total_rows,
        )

        return UpdateResult(
            symbol=symbol,
            success=True,
            gaps_filled=len(gaps),
            total_rows=total_rows,
        )

    def update_multiple(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> BatchResult:
        """Update multiple symbols incrementally.

        Continues updating even if individual symbols fail, collecting
        results and errors for each symbol.

        Args:
            symbols: List of canonical symbols
            start: Start of requested range (inclusive)
            end: End of requested range (inclusive)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            BatchResult containing UpdateResult for each symbol with
            success/failure status and update counts.
        """
        logger.info(
            "starting batch incremental update symbol_count=%d",
            len(symbols),
        )

        results: list[UpdateResult] = []

        for symbol in symbols:
            try:
                result = self.update(symbol, start, end, timeframe)
                results.append(result)

            except DataError as e:
                logger.error(
                    "failed to update symbol symbol=%s error=%s",
                    symbol,
                    str(e),
                )
                results.append(UpdateResult(symbol=symbol, success=False, error=str(e)))

        # Count successes
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes

        logger.info(
            "batch incremental update complete total=%d successes=%d failures=%d",
            len(symbols),
            successes,
            failures,
        )

        return BatchResult(
            total=len(results),
            succeeded=successes,
            failed=failures,
            results=results,
        )

    def update_to_now(
        self,
        symbol: str,
        lookback_days: int = 30,
        timeframe: str = "1d",
    ) -> UpdateResult:
        """Update data to current date.

        If existing data exists, fetches from the last date to now.
        If no data exists, fetches from lookback_days ago to now.

        Args:
            symbol: Canonical symbol (e.g., "EUR_USD")
            lookback_days: Days to look back if no data exists (default: 30)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            UpdateResult with gaps_filled and total_rows on success,
            or error message on failure.
        """
        storage_key = f"{self._asset_class}/{symbol}"
        interval = self._timeframe_to_timedelta(timeframe)

        # Check for existing data
        date_range = self._store.get_date_range(storage_key)

        if date_range is not None:
            _, existing_end = date_range
            interval_days = max(1, interval.days)
            start = existing_end + timedelta(days=interval_days)
        else:
            # No existing data, use lookback
            logger.info(
                "no existing data, using lookback symbol=%s lookback_days=%d",
                symbol,
                lookback_days,
            )
            start = date.today() - timedelta(days=lookback_days)

        end = date.today()

        if start > end:
            logger.info("data already up to date symbol=%s", symbol)
            return UpdateResult(
                symbol=symbol, success=True, gaps_filled=0, total_rows=0
            )

        logger.info(
            "updating to now symbol=%s start=%s end=%s",
            symbol,
            start.isoformat(),
            end.isoformat(),
        )

        try:
            return self.update(symbol, start, end, timeframe)

        except DataError as e:
            logger.error(
                "failed to update to now symbol=%s error=%s",
                symbol,
                str(e),
            )
            return UpdateResult(symbol=symbol, success=False, error=str(e))

    def _timeframe_to_timedelta(self, timeframe: str) -> timedelta:
        """Map timeframe string to timedelta.

        Args:
            timeframe: Candle timeframe string

        Returns:
            Corresponding timedelta

        Raises:
            ValueError: If timeframe not supported
        """
        if timeframe not in self._TIMEFRAME_TO_DELTA:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {list(self._TIMEFRAME_TO_DELTA.keys())}"
            )
        return self._TIMEFRAME_TO_DELTA[timeframe]

    def detect_internal_gaps(
        self,
        df: pl.DataFrame,
        timeframe: str = "1d",
        skip_weekends: bool = False,
    ) -> list[tuple[datetime, datetime]]:
        """Detect gaps within existing data (missing bars between timestamps).

        Analyzes a DataFrame to find missing bars based on expected timeframe spacing.

        Args:
            df: DataFrame with 'timestamp' column
            timeframe: Expected candle timeframe (e.g., "1m", "1h", "1d")
            skip_weekends: If True, ignore gaps that span weekends (for forex)

        Returns:
            List of (gap_start, gap_end) datetime tuples representing missing ranges.
            Empty list if no gaps detected.
        """
        if len(df) <= 1:
            return []

        interval = self._timeframe_to_timedelta(timeframe)

        # Sort by timestamp
        df_sorted = df.sort("timestamp")

        # Calculate time differences
        df_with_diff = df_sorted.with_columns([
            pl.col("timestamp").diff().alias("_time_diff")
        ])

        # Find rows where gap exceeds expected interval (with 10% tolerance)
        tolerance_factor = 1.1
        expected_seconds = interval.total_seconds()
        gap_threshold = expected_seconds * tolerance_factor

        gaps: list[tuple[datetime, datetime]] = []

        for row in df_with_diff.iter_rows(named=True):
            time_diff = row["_time_diff"]
            if time_diff is None:
                continue

            diff_seconds = time_diff.total_seconds()

            # Check if this represents a gap
            if diff_seconds > gap_threshold:
                current_ts = row["timestamp"]
                prev_ts = current_ts - time_diff

                # Calculate actual gap boundaries
                gap_start = prev_ts + interval
                gap_end = current_ts - interval

                # Only add if gap_end >= gap_start (at least one missing bar)
                if gap_end < gap_start:
                    gap_end = gap_start

                # Skip weekend gaps if requested
                if skip_weekends and self._is_weekend_gap(prev_ts, current_ts):
                    continue

                gaps.append((gap_start, gap_end))

        if gaps:
            logger.info(
                "internal gaps detected gap_count=%d",
                len(gaps),
            )

        return gaps

    def _is_weekend_gap(self, start: datetime, end: datetime) -> bool:
        """Check if a gap spans a weekend.

        Args:
            start: Gap start timestamp
            end: Gap end timestamp

        Returns:
            True if the gap spans from Friday to Monday (weekend)
        """
        # Friday is weekday 4, Saturday is 5, Sunday is 6, Monday is 0
        start_weekday = start.weekday()
        end_weekday = end.weekday()

        # Check if start is Friday/Saturday and end is Sunday/Monday
        is_friday_to_monday = (
            start_weekday in (4, 5)  # Friday or Saturday
            and end_weekday in (0, 6)  # Monday or Sunday
        )

        # Also check time duration (weekend gaps are typically 48-72 hours)
        duration_hours = (end - start).total_seconds() / 3600
        is_weekend_duration = MIN_WEEKEND_GAP_HOURS <= duration_hours <= MAX_WEEKEND_GAP_HOURS

        return is_friday_to_monday and is_weekend_duration

    def backfill_gaps(
        self,
        symbol: str,
        gaps: list[tuple[datetime, datetime]],
        timeframe: str = "1d",
    ) -> UpdateResult:
        """Backfill detected gaps by fetching missing data.

        Args:
            symbol: Canonical symbol (e.g., "EUR_USD")
            gaps: List of (start, end) datetime tuples from detect_internal_gaps
            timeframe: Candle timeframe (default: "1d")

        Returns:
            UpdateResult with gaps_filled and total_rows on success,
            or error message on failure.
        """
        if not gaps:
            return UpdateResult(
                symbol=symbol, success=True, gaps_filled=0, total_rows=0
            )

        logger.info(
            "starting gap backfill symbol=%s gap_count=%d",
            symbol,
            len(gaps),
        )

        total_rows = 0

        try:
            for gap_start, gap_end in gaps:
                # Convert datetime to date for fetch_and_store
                start_date = gap_start.date()
                end_date = gap_end.date()

                logger.info(
                    "backfilling gap symbol=%s gap_start=%s gap_end=%s",
                    symbol,
                    gap_start.isoformat(),
                    gap_end.isoformat(),
                )

                rows = self._fetcher.fetch_and_store(
                    symbol, start_date, end_date, timeframe
                )
                total_rows += rows

            logger.info(
                "gap backfill complete symbol=%s gaps_filled=%d total_rows=%d",
                symbol,
                len(gaps),
                total_rows,
            )

            return UpdateResult(
                symbol=symbol,
                success=True,
                gaps_filled=len(gaps),
                total_rows=total_rows,
            )

        except DataError as e:
            logger.error(
                "failed to backfill gaps symbol=%s error=%s",
                symbol,
                str(e),
            )
            return UpdateResult(symbol=symbol, success=False, error=str(e))

"""Data fetcher orchestration for pulling data from providers and storing it.

This module coordinates fetching OHLCV data from providers and persisting it
to storage, handling retries, errors, and batch operations.

Design Principles:
- SRP: Focused on orchestrating data flow between providers and storage
- DRY: Centralized retry and error handling logic
- KISS: Simple, clear interface for data fetching

Example:
    from liq.data.providers.oanda import OandaProvider
    from liq.store import ParquetStore
    from liq.data.fetcher import DataFetcher

    provider = OandaProvider(api_key="...", account_id="...", environment="practice")
    store = ParquetStore(data_root="./data")

    fetcher = DataFetcher(provider=provider, store=store, asset_class="forex")

    # Fetch and store data
    count = fetcher.fetch_and_store(
        symbol="EUR_USD",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        timeframe="1h"
    )
"""

import logging
from datetime import date

import polars as pl

from liq.core import BatchResult, FetchResult
from liq.data.exceptions import DataError, RateLimitError
from liq.data.policies import POLICIES
from liq.data.providers.base import BaseProvider
from liq.data.qa import validate_ohlc
from liq.data.rate_limiter import RateLimiter
from liq.data.retry import retry
from liq.store import key_builder
from liq.store.protocols import TimeSeriesStore

logger = logging.getLogger(__name__)


class DataFetcher:
    """Orchestrates data fetching from providers to storage.

    Handles the complete workflow of fetching OHLCV data from a provider,
    converting it to the required format, and persisting it to storage.

    Provides retry logic for transient failures like rate limits, and
    batch fetching capabilities for multiple symbols.
    """

    def __init__(
        self,
        provider: BaseProvider,
        store: TimeSeriesStore,
        asset_class: str = "forex",
    ) -> None:
        """Initialize DataFetcher with provider and storage.

        Args:
            provider: Data provider instance (OANDA, Binance, etc.)
            store: Storage backend instance (ParquetStore, etc.)
            asset_class: Asset class for the data (default: "forex")
        """
        self._provider = provider
        self._store = store
        self._asset_class = asset_class
        policy = POLICIES.get(provider.name, None)
        self._rate_limiter = RateLimiter(
            requests_per_minute=policy.requests_per_minute if policy else None,
            burst=policy.burst if policy else None,
        )

    @property
    def provider(self) -> BaseProvider:
        """Get the data provider."""
        return self._provider

    @property
    def store(self) -> TimeSeriesStore:
        """Get the storage backend."""
        return self._store

    @retry(
        max_retries=3,
        initial_delay=1.0,
        backoff_multiplier=2.0,
        retryable_exceptions=(RateLimitError,),
    )
    def fetch_and_store(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> int:
        """Fetch OHLCV data from provider and store it.

        This method handles the complete workflow:
        1. Fetch data from provider
        2. Add metadata columns (symbol, provider, asset_class)
        3. Store to storage backend using key format: {asset_class}/{symbol}

        Automatically retries on RateLimitError with exponential backoff.

        Args:
            symbol: Canonical symbol (e.g., "EUR_USD", "BTC_USDT")
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            Number of rows stored

        Raises:
            ProviderError: If provider fetch fails
            StorageError: If storage write fails
            RateLimitError: If rate limit persists after retries
        """
        logger.info(
            "fetching data symbol=%s start=%s end=%s timeframe=%s provider=%s",
            symbol,
            start.isoformat(),
            end.isoformat(),
            timeframe,
            self._provider.name,
        )

        # Rate limit enforcement
        self._rate_limiter.acquire()

        # Fetch data from provider
        df = self._provider.fetch_bars(symbol, start, end, timeframe=timeframe)

        if df.is_empty():
            logger.info("no data returned symbol=%s", symbol)
            return 0

        # Normalize timezone to UTC
        if "timestamp" not in df.columns:
            raise ValueError("provider returned data without timestamp column")
        df = df.with_columns(
            pl.col("timestamp")
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone("UTC")
            .alias("timestamp")
        )

        # Add metadata columns
        df = df.with_columns(
            [
                pl.lit(symbol).alias("symbol"),
                pl.lit(self._provider.name).alias("provider"),
                pl.lit(self._asset_class).alias("asset_class"),
            ]
        )

        # Validate before write
        validate_ohlc(df)

        # Generate storage key via liq-store
        storage_key = f"{self._provider.name}/{key_builder.bars(symbol, timeframe)}"

        # Store to storage backend
        self._store.write(storage_key, df, mode="append")

        logger.info(
            "stored data symbol=%s rows=%d provider=%s",
            symbol,
            len(df),
            self._provider.name,
        )

        return len(df)

    def fetch_multiple(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> BatchResult:
        """Fetch and store data for multiple symbols.

        Continues fetching even if individual symbols fail, collecting
        results and errors for each symbol.

        Args:
            symbols: List of canonical symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            BatchResult containing FetchResult for each symbol with
            success/failure status and counts.
        """
        logger.info(
            "fetching multiple symbols symbol_count=%d provider=%s",
            len(symbols),
            self._provider.name,
        )

        results: list[FetchResult] = []

        for symbol in symbols:
            try:
                count = self.fetch_and_store(symbol, start, end, timeframe)
                results.append(FetchResult(symbol=symbol, success=True, count=count))

            except DataError as e:
                logger.error(
                    "failed to fetch symbol symbol=%s error=%s provider=%s",
                    symbol,
                    str(e),
                    self._provider.name,
                )
                results.append(FetchResult(symbol=symbol, success=False, error=str(e)))

        # Count successes
        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes

        logger.info(
            "batch fetch complete total=%d successes=%d failures=%d",
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

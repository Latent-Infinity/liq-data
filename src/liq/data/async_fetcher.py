"""Async data fetcher for concurrent provider pulls.

This module provides asynchronous data fetching with concurrency control
and retry logic for handling multiple symbols efficiently.

Design Principles:
- SRP: Focused on async orchestration of data fetching
- DRY: Reuses core fetching logic with async wrappers
- KISS: Simple async interface with sensible defaults

Example:
    from liq.data.providers.binance import BinanceProvider
    from liq.store import ParquetStore
    from liq.data.async_fetcher import AsyncDataFetcher

    provider = BinanceProvider()
    store = ParquetStore(data_root="./data")

    fetcher = AsyncDataFetcher(
        provider=provider,
        store=store,
        asset_class="crypto",
        max_concurrency=5
    )

    # Fetch multiple symbols concurrently
    results = await fetcher.fetch_multiple(
        symbols=["BTC_USDT", "ETH_USDT", "SOL_USDT"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        timeframe="1h"
    )
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import polars as pl

from liq.core import BatchResult, FetchResult
from liq.data.exceptions import DataError, RateLimitError
from liq.data.providers.base import BaseProvider
from liq.store.protocols import TimeSeriesStore

logger = logging.getLogger(__name__)


@dataclass
class AsyncRetryPolicy:
    """Configuration for async retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        backoff: Multiplier for exponential backoff (default: 2.0)
    """

    max_retries: int = 3
    base_delay: float = 1.0
    backoff: float = 2.0


class AsyncDataFetcher:
    """Async fetcher that wraps sync providers/storage with concurrency and retries.

    Provides concurrent data fetching with configurable concurrency limits
    and exponential backoff retry logic for rate limit handling.
    """

    def __init__(
        self,
        provider: BaseProvider,
        store: TimeSeriesStore,
        asset_class: str = "forex",
        max_concurrency: int = 5,
        retry_policy: AsyncRetryPolicy | None = None,
    ) -> None:
        """Initialize AsyncDataFetcher.

        Args:
            provider: Data provider instance (OANDA, Binance, etc.)
            store: Storage backend instance (ParquetStore, etc.)
            asset_class: Asset class for the data (default: "forex")
            max_concurrency: Maximum concurrent fetch operations (default: 5)
            retry_policy: Retry configuration (default: AsyncRetryPolicy())
        """
        self._provider = provider
        self._store = store
        self._asset_class = asset_class
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._retry_policy = retry_policy or AsyncRetryPolicy()

    @property
    def provider(self) -> BaseProvider:
        """Get the data provider."""
        return self._provider

    @property
    def store(self) -> TimeSeriesStore:
        """Get the storage backend."""
        return self._store

    async def fetch_and_store(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> int:
        """Fetch OHLCV data asynchronously and store it.

        Uses a semaphore to limit concurrent operations and retries
        on rate limit errors with exponential backoff.

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
        async with self._semaphore:
            df = await self._fetch_with_retries(symbol, start, end, timeframe)

            if df.is_empty():
                logger.info(
                    "no data returned symbol=%s provider=%s",
                    symbol,
                    self._provider.name,
                )
                return 0

            # Add metadata columns
            df = df.with_columns(
                [
                    pl.lit(symbol).alias("symbol"),
                    pl.lit(self._provider.name).alias("provider"),
                    pl.lit(self._asset_class).alias("asset_class"),
                ]
            )

            # Generate storage key
            storage_key = f"{self._asset_class}/{symbol}"

            # Store asynchronously
            await asyncio.to_thread(self._store.write, storage_key, df, "append")

            logger.info(
                "stored data async symbol=%s rows=%d provider=%s",
                symbol,
                len(df),
                self._provider.name,
            )

            return len(df)

    async def fetch_multiple(
        self,
        symbols: Iterable[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> BatchResult:
        """Fetch multiple symbols concurrently.

        Creates concurrent tasks for each symbol and gathers results.

        Args:
            symbols: Iterable of canonical symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Candle timeframe (default: "1d")

        Returns:
            BatchResult containing FetchResult for each symbol with
            success/failure status and counts.
        """
        tasks = []
        for symbol in symbols:
            tasks.append(
                asyncio.create_task(
                    self._fetch_symbol(symbol, start, end, timeframe)
                )
            )

        results: list[FetchResult] = await asyncio.gather(*tasks)

        successes = sum(1 for r in results if r.success)
        failures = len(results) - successes

        return BatchResult(
            total=len(results),
            succeeded=successes,
            failed=failures,
            results=results,
        )

    async def _fetch_symbol(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str,
    ) -> FetchResult:
        """Fetch a single symbol with error handling.

        Args:
            symbol: Canonical symbol
            start: Start date
            end: End date
            timeframe: Candle timeframe

        Returns:
            FetchResult with success status and count or error message.
        """
        try:
            count = await self.fetch_and_store(symbol, start, end, timeframe)
            return FetchResult(symbol=symbol, success=True, count=count)
        except DataError as exc:
            logger.error(
                "async fetch failed symbol=%s error=%s",
                symbol,
                str(exc),
            )
            return FetchResult(symbol=symbol, success=False, error=str(exc))

    async def _fetch_with_retries(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str,
    ) -> pl.DataFrame:
        """Fetch data with retry logic for rate limits.

        Args:
            symbol: Canonical symbol
            start: Start date
            end: End date
            timeframe: Candle timeframe

        Returns:
            Polars DataFrame with OHLCV data

        Raises:
            RateLimitError: If rate limit persists after retries
            ProviderError: If provider fetch fails for other reasons
        """
        attempts = 0

        while True:
            try:
                return await asyncio.to_thread(
                    self._provider.fetch_bars, symbol, start, end, timeframe
                )
            except RateLimitError as exc:
                attempts += 1
                if attempts >= self._retry_policy.max_retries:
                    logger.error(
                        "async retry exhausted symbol=%s provider=%s error=%s",
                        symbol,
                        self._provider.name,
                        str(exc),
                    )
                    raise

                delay = self._retry_policy.base_delay * (
                    self._retry_policy.backoff ** (attempts - 1)
                )
                logger.warning(
                    "rate limit hit; backing off symbol=%s provider=%s attempt=%d delay=%.2f",
                    symbol,
                    self._provider.name,
                    attempts,
                    delay,
                )
                await asyncio.sleep(delay)

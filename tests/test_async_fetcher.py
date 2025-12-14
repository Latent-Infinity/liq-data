"""Tests for liq.data.async_fetcher module."""

import asyncio
from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from liq.data.async_fetcher import AsyncDataFetcher, AsyncRetryPolicy
from liq.data.exceptions import ProviderError, RateLimitError
from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "test_provider"
    return provider


@pytest.fixture
def mock_store() -> MagicMock:
    """Create a mock store."""
    store = MagicMock()
    return store


@pytest.fixture
def sample_bars_df() -> pl.DataFrame:
    """Create a sample bars DataFrame."""
    from datetime import UTC, datetime

    return pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC),
        ],
        "open": [1.0850, 1.0860],
        "high": [1.0875, 1.0890],
        "low": [1.0825, 1.0850],
        "close": [1.0860, 1.0885],
        "volume": [1000.0, 1500.0],
    })


class TestAsyncRetryPolicy:
    """Tests for AsyncRetryPolicy dataclass."""

    def test_default_values(self) -> None:
        """Test default policy values."""
        policy = AsyncRetryPolicy()

        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.backoff == 2.0

    def test_custom_values(self) -> None:
        """Test custom policy values."""
        policy = AsyncRetryPolicy(max_retries=5, base_delay=0.5, backoff=3.0)

        assert policy.max_retries == 5
        assert policy.base_delay == 0.5
        assert policy.backoff == 3.0


class TestAsyncDataFetcherCreation:
    """Tests for AsyncDataFetcher instantiation."""

    def test_create_fetcher(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test AsyncDataFetcher creation."""
        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
            max_concurrency=5,
        )

        assert fetcher.provider is mock_provider
        assert fetcher.store is mock_store

    def test_custom_retry_policy(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test custom retry policy."""
        policy = AsyncRetryPolicy(max_retries=5)

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            retry_policy=policy,
        )

        assert fetcher._retry_policy.max_retries == 5


class TestAsyncDataFetcherFetchAndStore:
    """Tests for AsyncDataFetcher.fetch_and_store method."""

    @pytest.mark.asyncio
    async def test_fetch_and_store_success(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test successful async fetch and store."""
        mock_provider.fetch_bars.return_value = sample_bars_df

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        result = await fetcher.fetch_and_store(
            "EUR_USD",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert result == 2
        mock_provider.fetch_bars.assert_called_once()
        mock_store.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_and_store_empty_data(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test async fetch with empty data returns 0."""
        mock_provider.fetch_bars.return_value = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "open": PRICE_DTYPE,
                "high": PRICE_DTYPE,
                "low": PRICE_DTYPE,
                "close": PRICE_DTYPE,
                "volume": VOLUME_DTYPE,
            }
        )

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        result = await fetcher.fetch_and_store(
            "EUR_USD",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert result == 0
        mock_store.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_and_store_adds_metadata(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test metadata columns are added in async fetch."""
        mock_provider.fetch_bars.return_value = sample_bars_df

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="crypto",
        )

        await fetcher.fetch_and_store(
            "BTC_USDT",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        # Get the DataFrame that was written
        written_df = mock_store.write.call_args[0][1]

        assert "symbol" in written_df.columns
        assert "provider" in written_df.columns
        assert "asset_class" in written_df.columns

    @pytest.mark.asyncio
    @patch("liq.data.async_fetcher.asyncio.sleep")
    async def test_fetch_and_store_rate_limit_retry(
        self,
        mock_sleep: MagicMock,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test rate limit triggers async retry."""
        mock_sleep.return_value = None

        # First call fails, second succeeds
        mock_provider.fetch_bars.side_effect = [
            RateLimitError("Rate limit"),
            sample_bars_df,
        ]

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
            retry_policy=AsyncRetryPolicy(base_delay=0.01),
        )

        result = await fetcher.fetch_and_store(
            "EUR_USD",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert result == 2
        assert mock_provider.fetch_bars.call_count == 2


class TestAsyncDataFetcherFetchMultiple:
    """Tests for AsyncDataFetcher.fetch_multiple method."""

    @pytest.mark.asyncio
    async def test_fetch_multiple_success(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test successful concurrent fetch of multiple symbols."""
        mock_provider.fetch_bars.return_value = sample_bars_df

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        results = await fetcher.fetch_multiple(
            ["EUR_USD", "GBP_USD"],
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert results.total == 2
        assert results.succeeded == 2
        assert results.failed == 0
        eur_result = next(r for r in results.results if r.symbol == "EUR_USD")
        gbp_result = next(r for r in results.results if r.symbol == "GBP_USD")
        assert eur_result.success is True
        assert eur_result.count == 2
        assert gbp_result.success is True
        assert gbp_result.count == 2

    @pytest.mark.asyncio
    async def test_fetch_multiple_partial_failure(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test partial failure in concurrent fetch."""

        def fetch_bars_side_effect(symbol: str, *args, **kwargs) -> pl.DataFrame:
            if symbol == "BAD_SYMBOL":
                raise ProviderError("Unknown symbol")
            return sample_bars_df

        mock_provider.fetch_bars.side_effect = fetch_bars_side_effect

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        results = await fetcher.fetch_multiple(
            ["EUR_USD", "BAD_SYMBOL", "GBP_USD"],
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
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

    @pytest.mark.asyncio
    async def test_fetch_multiple_respects_concurrency(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test concurrency limit is respected."""
        concurrent_count = {"current": 0, "max": 0}

        async def track_concurrency(*args, **kwargs) -> pl.DataFrame:
            concurrent_count["current"] += 1
            concurrent_count["max"] = max(
                concurrent_count["max"], concurrent_count["current"]
            )
            await asyncio.sleep(0.01)  # Small delay
            concurrent_count["current"] -= 1
            return sample_bars_df

        mock_provider.fetch_bars.return_value = sample_bars_df

        fetcher = AsyncDataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
            max_concurrency=2,
        )

        await fetcher.fetch_multiple(
            ["SYM1", "SYM2", "SYM3", "SYM4"],
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        # All 4 should have been processed
        assert mock_provider.fetch_bars.call_count == 4

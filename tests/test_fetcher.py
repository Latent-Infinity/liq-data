"""Tests for liq.data.fetcher module."""

from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from liq.data.exceptions import ProviderError, RateLimitError
from liq.data.fetcher import DataFetcher


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


class TestDataFetcherCreation:
    """Tests for DataFetcher instantiation."""

    def test_create_fetcher(self, mock_provider: MagicMock, mock_store: MagicMock) -> None:
        """Test DataFetcher creation."""
        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        assert fetcher.provider is mock_provider
        assert fetcher.store is mock_store

    def test_default_asset_class(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test default asset_class is forex."""
        fetcher = DataFetcher(provider=mock_provider, store=mock_store)

        # Access private attribute to verify
        assert fetcher._asset_class == "forex"


class TestDataFetcherFetchAndStore:
    """Tests for DataFetcher.fetch_and_store method."""

    def test_fetch_and_store_success(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test successful fetch and store."""
        mock_provider.fetch_bars.return_value = sample_bars_df

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        result = fetcher.fetch_and_store(
            "EUR_USD",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert result == 2
        mock_provider.fetch_bars.assert_called_once_with(
            "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )
        mock_store.write.assert_called_once()

        # Verify storage key format
        call_args = mock_store.write.call_args
        assert call_args[0][0] == "forex/EUR_USD"
        assert call_args[1]["mode"] == "append"

    def test_fetch_and_store_empty_data(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test fetch with empty data returns 0."""
        mock_provider.fetch_bars.return_value = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        result = fetcher.fetch_and_store(
            "EUR_USD",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert result == 0
        mock_store.write.assert_not_called()

    def test_fetch_and_store_adds_metadata(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test metadata columns are added."""
        mock_provider.fetch_bars.return_value = sample_bars_df

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="crypto",
        )

        fetcher.fetch_and_store(
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
        assert written_df["symbol"][0] == "BTC_USDT"
        assert written_df["provider"][0] == "test_provider"
        assert written_df["asset_class"][0] == "crypto"

    def test_fetch_and_store_normalizes_timezone(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Ensure timestamps are UTC tz-aware."""
        # make timestamps naive
        naive_df = sample_bars_df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        mock_provider.fetch_bars.return_value = naive_df

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        fetcher.fetch_and_store(
            "EUR_USD",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        written_df = mock_store.write.call_args[0][1]
        assert str(written_df.schema["timestamp"].time_unit) == "us"
        assert written_df.schema["timestamp"].time_zone == "UTC"

    def test_fetch_and_store_provider_error(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test provider error is propagated."""
        mock_provider.fetch_bars.side_effect = ProviderError("API error")

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        with pytest.raises(ProviderError, match="API error"):
            fetcher.fetch_and_store(
                "EUR_USD",
                date(2024, 1, 15),
                date(2024, 1, 15),
                timeframe="1h",
            )

    @patch("liq.data.retry.time.sleep")  # Patch sleep to speed up test
    def test_fetch_and_store_rate_limit_retry(
        self,
        mock_sleep: MagicMock,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test rate limit triggers retry."""
        # First call fails, second succeeds
        mock_provider.fetch_bars.side_effect = [
            RateLimitError("Rate limit"),
            sample_bars_df,
        ]

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        result = fetcher.fetch_and_store(
            "EUR_USD",
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert result == 2
        assert mock_provider.fetch_bars.call_count == 2


class TestDataFetcherFetchMultiple:
    """Tests for DataFetcher.fetch_multiple method."""

    def test_fetch_multiple_success(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test successful fetch of multiple symbols."""
        mock_provider.fetch_bars.return_value = sample_bars_df

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        results = fetcher.fetch_multiple(
            ["EUR_USD", "GBP_USD"],
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert len(results) == 2
        assert results["EUR_USD"]["success"] is True
        assert results["EUR_USD"]["count"] == 2
        assert results["GBP_USD"]["success"] is True
        assert results["GBP_USD"]["count"] == 2

    def test_fetch_multiple_partial_failure(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_bars_df: pl.DataFrame,
    ) -> None:
        """Test partial failure in multiple fetch."""

        def fetch_bars_side_effect(symbol: str, *args, **kwargs) -> pl.DataFrame:
            if symbol == "BAD_SYMBOL":
                raise ProviderError("Unknown symbol")
            return sample_bars_df

        mock_provider.fetch_bars.side_effect = fetch_bars_side_effect

        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        results = fetcher.fetch_multiple(
            ["EUR_USD", "BAD_SYMBOL", "GBP_USD"],
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert results["EUR_USD"]["success"] is True
        assert results["BAD_SYMBOL"]["success"] is False
        assert "error" in results["BAD_SYMBOL"]
        assert results["GBP_USD"]["success"] is True

    def test_fetch_multiple_empty_list(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test empty symbol list returns empty results."""
        fetcher = DataFetcher(
            provider=mock_provider,
            store=mock_store,
            asset_class="forex",
        )

        results = fetcher.fetch_multiple(
            [],
            date(2024, 1, 15),
            date(2024, 1, 15),
            timeframe="1h",
        )

        assert results == {}
        mock_provider.fetch_bars.assert_not_called()

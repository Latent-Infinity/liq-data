"""Tests for liq.data.instruments module."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from liq.data.instruments import InstrumentSync


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
def sample_instruments_df() -> pl.DataFrame:
    """Create a sample instruments DataFrame."""
    return pl.DataFrame({
        "symbol": ["EUR_USD", "GBP_USD", "USD_JPY"],
        "name": ["Euro/US Dollar", "British Pound/US Dollar", "US Dollar/Japanese Yen"],
        "asset_class": ["forex", "forex", "forex"],
        "type": ["CURRENCY", "CURRENCY", "CURRENCY"],
    })


class TestInstrumentSyncCreation:
    """Tests for InstrumentSync instantiation."""

    def test_create_sync(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test InstrumentSync creation."""
        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        assert sync.provider is mock_provider
        assert sync.store is mock_store


class TestInstrumentSyncFetchInstruments:
    """Tests for InstrumentSync.fetch_instruments method."""

    def test_fetch_instruments_success(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_instruments_df: pl.DataFrame,
    ) -> None:
        """Test successful instrument fetch."""
        mock_provider.list_instruments.return_value = sample_instruments_df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.fetch_instruments("forex")

        assert len(result) == 3
        assert "symbol" in result.columns
        assert "provider" in result.columns
        mock_provider.list_instruments.assert_called_once_with("forex")

    def test_fetch_instruments_empty(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test fetch with empty result."""
        mock_provider.list_instruments.return_value = pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "name": pl.Utf8,
            }
        )

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.fetch_instruments("forex")

        assert result.is_empty()

    def test_fetch_instruments_adds_provider(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_instruments_df: pl.DataFrame,
    ) -> None:
        """Test provider column is added."""
        mock_provider.list_instruments.return_value = sample_instruments_df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.fetch_instruments("forex")

        assert "provider" in result.columns
        assert result["provider"][0] == "test_provider"

    def test_fetch_instruments_generates_name(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test name column is generated from symbol if not present."""
        df = pl.DataFrame({
            "symbol": ["EUR_USD", "GBP_USD"],
        })
        mock_provider.list_instruments.return_value = df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.fetch_instruments("forex")

        assert "name" in result.columns
        assert result["name"][0] == "EUR/USD"
        assert result["name"][1] == "GBP/USD"


class TestInstrumentSyncSyncInstruments:
    """Tests for InstrumentSync.sync_instruments method."""

    def test_sync_instruments_success(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_instruments_df: pl.DataFrame,
    ) -> None:
        """Test successful instrument sync."""
        mock_provider.list_instruments.return_value = sample_instruments_df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        count = sync.sync_instruments("forex")

        assert count == 3
        mock_store.write.assert_called_once()

        # Verify storage key
        call_args = mock_store.write.call_args
        assert call_args[0][0] == "instruments/test_provider"
        assert call_args[1]["mode"] == "overwrite"

    def test_sync_instruments_empty(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test sync with empty instruments."""
        mock_provider.list_instruments.return_value = pl.DataFrame(
            schema={"symbol": pl.Utf8}
        )

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        count = sync.sync_instruments("forex")

        assert count == 0
        mock_store.write.assert_not_called()

    def test_sync_instruments_without_asset_class(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_instruments_df: pl.DataFrame,
    ) -> None:
        """Test sync without specifying asset class."""
        mock_provider.list_instruments.return_value = sample_instruments_df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        count = sync.sync_instruments()

        assert count == 3
        mock_provider.list_instruments.assert_called_once_with(None)


class TestInstrumentSyncGetInstruments:
    """Tests for InstrumentSync.get_instruments method."""

    def test_get_instruments_exists(
        self,
        mock_provider: MagicMock,
        mock_store: MagicMock,
        sample_instruments_df: pl.DataFrame,
    ) -> None:
        """Test get instruments when data exists."""
        mock_store.exists.return_value = True
        mock_store.read.return_value = sample_instruments_df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.get_instruments()

        assert len(result) == 3
        mock_store.exists.assert_called_once_with("instruments/test_provider")
        mock_store.read.assert_called_once_with("instruments/test_provider")

    def test_get_instruments_not_synced(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test get instruments when not synced returns empty."""
        mock_store.exists.return_value = False

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.get_instruments()

        assert result.is_empty()
        mock_store.read.assert_not_called()


class TestInstrumentSyncNormalization:
    """Tests for instrument normalization."""

    def test_normalize_preserves_extra_columns(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test normalization preserves extra columns."""
        df = pl.DataFrame({
            "symbol": ["EUR_USD"],
            "name": ["Euro/USD"],
            "extra_field": ["value"],
            "another_field": [123],
        })
        mock_provider.list_instruments.return_value = df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.fetch_instruments("forex")

        # Required columns should be present
        assert "symbol" in result.columns
        assert "provider" in result.columns
        assert "asset_class" in result.columns
        assert "name" in result.columns

        # Extra columns should be preserved
        assert "extra_field" in result.columns
        assert "another_field" in result.columns

    def test_normalize_handles_missing_columns(
        self, mock_provider: MagicMock, mock_store: MagicMock
    ) -> None:
        """Test normalization handles missing optional columns."""
        df = pl.DataFrame({
            "symbol": ["EUR_USD"],
        })
        mock_provider.list_instruments.return_value = df

        sync = InstrumentSync(provider=mock_provider, store=mock_store)

        result = sync.fetch_instruments("forex")

        # Should have added provider and asset_class
        assert "symbol" in result.columns
        assert "provider" in result.columns
        assert "asset_class" in result.columns
        assert "name" in result.columns  # Generated from symbol

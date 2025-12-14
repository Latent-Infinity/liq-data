"""Instrument synchronization for fetching and storing instrument catalogs.

This module provides the InstrumentSync class that fetches available instruments
from providers and stores them in the storage backend.

Design Principles:
- SRP: Focused on instrument catalog synchronization
- DIP: Depends on BaseProvider and TimeSeriesStore abstractions
- KISS: Simple fetch-normalize-store workflow

Features:
- Fetch instruments from providers
- Normalize instrument data to standard schema
- Store instrument catalog
- Generate human-readable names

Example:
    from liq.data.instruments import InstrumentSync
    from liq.data.providers.oanda import OandaProvider
    from liq.store import ParquetStore

    provider = OandaProvider(api_key="...", account_id="...", environment="practice")
    store = ParquetStore(data_root="./data")

    sync = InstrumentSync(provider=provider, store=store)

    # Sync instruments for an asset class
    count = sync.sync_instruments("forex")
    print(f"Synced {count} instruments")
"""

import logging

import polars as pl

from liq.data.providers.base import BaseProvider
from liq.store.protocols import TimeSeriesStore

logger = logging.getLogger(__name__)


class InstrumentSync:
    """Synchronize instrument catalogs from providers to storage.

    Fetches available instruments from a provider and stores them
    with normalized metadata.
    """

    def __init__(self, provider: BaseProvider, store: TimeSeriesStore) -> None:
        """Initialize InstrumentSync.

        Args:
            provider: Provider instance to fetch instruments from
            store: Storage instance to write instruments to
        """
        self._provider = provider
        self._store = store

        logger.info("InstrumentSync initialized provider=%s", self._provider.name)

    @property
    def provider(self) -> BaseProvider:
        """Get the data provider."""
        return self._provider

    @property
    def store(self) -> TimeSeriesStore:
        """Get the storage backend."""
        return self._store

    def fetch_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        """Fetch instruments from provider and normalize to DataFrame.

        Args:
            asset_class: Asset class to fetch (e.g., "forex", "crypto")

        Returns:
            Polars DataFrame with normalized instrument metadata
        """
        logger.info(
            "Fetching instruments provider=%s asset_class=%s",
            self._provider.name,
            asset_class,
        )

        # Fetch from provider
        instruments_df = self._provider.list_instruments(asset_class)

        if instruments_df.is_empty():
            logger.warning("No instruments fetched provider=%s", self._provider.name)
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "provider": pl.Utf8,
                    "asset_class": pl.Utf8,
                    "name": pl.Utf8,
                }
            )

        # Normalize columns
        instruments_df = self._normalize_instruments(instruments_df, asset_class)

        logger.info(
            "Fetched instruments provider=%s count=%d",
            self._provider.name,
            len(instruments_df),
        )

        return instruments_df

    def _normalize_instruments(
        self,
        df: pl.DataFrame,
        asset_class: str | None,
    ) -> pl.DataFrame:
        """Normalize instrument data to standard schema.

        Args:
            df: Raw instrument DataFrame
            asset_class: Asset class

        Returns:
            Normalized DataFrame with standard columns
        """
        # Add provider column if not present
        if "provider" not in df.columns:
            df = df.with_columns([pl.lit(self._provider.name).alias("provider")])

        # Add asset_class if provided and not present
        if "asset_class" not in df.columns and asset_class is not None:
            df = df.with_columns([pl.lit(asset_class).alias("asset_class")])

        # Generate name if not present (replace _ with /)
        if "name" not in df.columns and "symbol" in df.columns:
            df = df.with_columns([pl.col("symbol").str.replace("_", "/").alias("name")])

        # Ensure required columns exist with defaults
        required_cols = ["symbol", "provider", "asset_class", "name"]
        available_cols = [col for col in required_cols if col in df.columns]

        # Add any additional columns that exist in the data
        extra_cols = [col for col in df.columns if col not in required_cols]
        select_cols = available_cols + extra_cols

        return df.select(select_cols)

    def sync_instruments(self, asset_class: str | None = None) -> int:
        """Sync instruments to storage.

        Fetches instruments from provider and writes them to storage
        using key format: instruments/{provider}

        Args:
            asset_class: Asset class to sync

        Returns:
            Number of instruments synced

        Raises:
            ProviderError: If provider fetch fails
            StorageError: If storage write fails
        """
        logger.info(
            "Starting instrument sync provider=%s asset_class=%s",
            self._provider.name,
            asset_class,
        )

        # Fetch instruments
        df = self.fetch_instruments(asset_class)

        if df.is_empty():
            logger.warning("No instruments to sync provider=%s", self._provider.name)
            return 0

        # Write to storage using key: instruments/{provider}
        storage_key = f"instruments/{self._provider.name}"
        self._store.write(storage_key, df, mode="overwrite")

        logger.info(
            "Instrument sync completed provider=%s count=%d",
            self._provider.name,
            len(df),
        )

        return len(df)

    def get_instruments(self) -> pl.DataFrame:
        """Get previously synced instruments from storage.

        Returns:
            DataFrame with instrument data or empty DataFrame if not synced

        Raises:
            StorageError: If storage read fails
        """
        storage_key = f"instruments/{self._provider.name}"

        if not self._store.exists(storage_key):
            logger.info(
                "No instruments in storage provider=%s",
                self._provider.name,
            )
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "provider": pl.Utf8,
                    "asset_class": pl.Utf8,
                    "name": pl.Utf8,
                }
            )

        return self._store.read(storage_key)

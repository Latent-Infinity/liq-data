"""DataService: Programmatic API for liq-data operations.

This module provides a Python API for data operations, wrapping CLI
functionality for programmatic access.

Design Principles:
    - SRP: DataService handles high-level data operations
    - OCP: New operations can be added without modifying existing methods
    - DIP: Depends on abstractions (DataProvider protocol), not implementations
    - DRY: Uses liq-store for all storage operations (single source of truth)

Example:
    from liq.data import DataService
    from datetime import date

    # Initialize service
    ds = DataService()

    # Fetch data from provider
    df = ds.fetch("oanda", "EUR_USD", date(2024, 1, 1), date(2024, 1, 31))

    # Load existing data
    df = ds.load("oanda", "EUR_USD", "1m")

    # List available data
    available = ds.list_symbols()

    # Validate data integrity
    result = ds.validate("oanda", "EUR_USD", "1m")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from liq.data.aggregation import aggregate_bars
from liq.data.gaps import detect_gaps
from liq.data.qa import validate_ohlc
from liq.data.settings import (
    LiqDataSettings,
    create_alpaca_provider,
    create_binance_provider,
    create_coinbase_provider,
    create_oanda_provider,
    create_polygon_provider,
    create_tradestation_provider,
    get_settings,
)
from liq.store import key_builder
from liq.store.parquet import ParquetStore

if TYPE_CHECKING:
    from liq.data.protocols import DataProvider

logger = logging.getLogger(__name__)


class DataService:
    """Programmatic API for liq-data operations.

    Provides a high-level Python interface for fetching, loading,
    validating, and managing market data.

    Attributes:
        settings: LiqDataSettings instance
        data_root: Path to data storage directory

    Example:
        >>> ds = DataService()
        >>> df = ds.load("oanda", "EUR_USD", "1m")
        >>> len(df)
        1000000
    """

    # Provider factory mapping
    _PROVIDER_FACTORIES = {
        "oanda": create_oanda_provider,
        "binance": create_binance_provider,
        "tradestation": create_tradestation_provider,
        "coinbase": create_coinbase_provider,
        "polygon": create_polygon_provider,
        "alpaca": create_alpaca_provider,
    }

    def __init__(
        self,
        settings: LiqDataSettings | None = None,
        data_root: Path | None = None,
    ) -> None:
        """Initialize DataService.

        Args:
            settings: Optional settings instance (uses get_settings() if not provided)
            data_root: Optional override for data storage directory
        """
        self._settings = settings or get_settings()
        self._data_root = data_root or self._settings.data_root
        self._store = ParquetStore(str(self._data_root))

    @property
    def settings(self) -> LiqDataSettings:
        """Get settings instance."""
        return self._settings

    @property
    def data_root(self) -> Path:
        """Get data root directory."""
        return self._data_root

    @property
    def store(self) -> ParquetStore:
        """Get the underlying ParquetStore instance."""
        return self._store

    def _storage_key(self, provider: str, symbol: str, timeframe: str) -> str:
        """Build storage key for bars via provider/key_builder."""
        return f"{provider}/{key_builder.bars(symbol, timeframe)}"

    def _get_provider(self, provider_name: str) -> DataProvider:
        """Get a provider instance by name.

        Args:
            provider_name: Provider name (e.g., "oanda", "binance")

        Returns:
            DataProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        factory = self._PROVIDER_FACTORIES.get(provider_name.lower())
        if factory is None:
            supported = ", ".join(self._PROVIDER_FACTORIES.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. Supported: {supported}"
            )
        return factory(self._settings)

    def load(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date | None = None,
        end: date | None = None,
        columns: list[str] | None = None,
        *,
        streaming: bool = False,
        batch_size: int | None = None,
    ) -> pl.DataFrame | Iterator[pl.DataFrame]:
        """Load data for a symbol from storage.

        Args:
            provider: Provider name (e.g., "oanda", "binance")
            symbol: Symbol name (e.g., "EUR_USD", "BTC_USDT")
            timeframe: Timeframe (e.g., "1m", "1h", "1d")
            start: Optional start date filter (inclusive)
            end: Optional end date filter (inclusive)
            columns: Optional column subset
            streaming: Use Polars streaming engine (memory-efficient)
            batch_size: If set, yield DataFrames in batches

        Returns:
            DataFrame with OHLCV data

        Raises:
            FileNotFoundError: If data doesn't exist

        Example:
            >>> ds = DataService()
            >>> df = ds.load("oanda", "EUR_USD", "1m")
        """
        storage_key = self._storage_key(provider, symbol, timeframe)
        if self._store.exists(storage_key) and timeframe != "1m":
            base_key = self._storage_key(provider, symbol, "1m")
            if start is None and end is None and self._store.exists(base_key):
                base_latest_df = self._store.read_latest(base_key, n=1)
                target_latest_df = self._store.read_latest(storage_key, n=1)
                if not base_latest_df.is_empty() and not target_latest_df.is_empty():
                    base_latest = base_latest_df["timestamp"][0]
                    target_latest = target_latest_df["timestamp"][0]
                    if isinstance(base_latest, datetime) and base_latest.tzinfo is None:
                        base_latest = base_latest.replace(tzinfo=UTC)
                    if isinstance(target_latest, datetime) and target_latest.tzinfo is None:
                        target_latest = target_latest.replace(tzinfo=UTC)
                    if base_latest > target_latest:
                        logger.info(
                            "Refreshing cached %s aggregate for %s/%s: base range %s-%s exceeds cached %s-%s",
                            timeframe,
                            provider,
                            symbol,
                            base_latest,
                            base_latest,
                            target_latest,
                            target_latest,
                        )
                        base_df = self._store.read(
                            base_key,
                            streaming=streaming,
                        )
                        aggregated = aggregate_bars(base_df, timeframe)
                        if not aggregated.is_empty():
                            self._store.write(storage_key, aggregated, mode="overwrite")
        if not self._store.exists(storage_key):
            if timeframe != "1m":
                base_key = self._storage_key(provider, symbol, "1m")
                if self._store.exists(base_key):
                    if batch_size is not None:
                        raise ValueError(
                            "batch_size is not supported for aggregated reads"
                        )
                    base_df = self._store.read(
                        base_key,
                        start=start,
                        end=end,
                        streaming=streaming,
                    )
                    aggregated = aggregate_bars(base_df, timeframe)
                    if not aggregated.is_empty():
                        self._store.write(storage_key, aggregated, mode="overwrite")
                    return aggregated
            raise FileNotFoundError(
                f"Data not found for {provider}/{symbol}/{timeframe}. "
                f"Run: ds.fetch('{provider}', '{symbol}', start, end, '{timeframe}')"
            )
        return self._store.read(
            storage_key,
            start=start,
            end=end,
            columns=columns,
            streaming=streaming,
            batch_size=batch_size,
        )

    def iter_batches(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date | None = None,
        end: date | None = None,
        columns: list[str] | None = None,
        *,
        batch_size: int = 100_000,
    ) -> Iterator[pl.DataFrame]:
        """Iterate over data in batches for memory-efficient processing."""
        result = self.load(
            provider,
            symbol,
            timeframe,
            start=start,
            end=end,
            columns=columns,
            batch_size=batch_size,
        )
        if isinstance(result, pl.DataFrame):
            yield result
            return
        yield from result

    def list_symbols(self, provider: str | None = None) -> list[dict[str, str]]:
        """List available data files.

        Args:
            provider: Optional filter by provider name

        Returns:
            List of dicts with provider, symbol, timeframe info

        Example:
            >>> ds = DataService()
            >>> ds.list_symbols()
            [{'provider': 'oanda', 'symbol': 'EUR_USD', 'timeframe': '1m'}, ...]
        """
        prefix = f"{provider}/" if provider else ""
        keys = self._store.list_keys(prefix=prefix)

        result = []
        for key in keys:
            parts = key.split("/")
            if len(parts) >= 4 and parts[2] == "bars":  # provider/symbol/bars/timeframe
                result.append({
                    "provider": parts[0],
                    "symbol": parts[1],
                    "timeframe": parts[3],
                })

        return result

    def fetch(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date | None = None,
        timeframe: str = "1m",
        save: bool = True,
        mode: str = "append",
    ) -> pl.DataFrame:
        """Fetch data from provider.

        Args:
            provider: Provider name (e.g., "oanda", "binance")
            symbol: Symbol name (e.g., "EUR_USD", "BTC_USDT")
            start: Start date
            end: End date (defaults to today)
            timeframe: Timeframe (default: "1m")
            save: Whether to save to storage (default: True)
            mode: Write mode - "append" (merge with existing) or "overwrite"

        Returns:
            DataFrame with fetched OHLCV data

        Example:
            >>> from datetime import date
            >>> ds = DataService()
            >>> df = ds.fetch("oanda", "EUR_USD", date(2024, 1, 1), date(2024, 1, 31))
        """
        if end is None:
            end = date.today()

        prov = self._get_provider(provider)
        df = prov.fetch_bars(symbol, start, end, timeframe=timeframe)
        validate_ohlc(df)

        if save:
            storage_key = self._storage_key(provider, symbol, timeframe)
            self._store.write(storage_key, df, mode=mode)

        return df

    def fetch_quotes(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date | None = None,
        save: bool = False,
        mode: str = "append",
    ) -> pl.DataFrame:
        """Fetch quotes when provider supports it."""
        if end is None:
            end = date.today()
        prov = self._get_provider(provider)
        if not hasattr(prov, "fetch_quotes"):
            raise ValueError(f"Provider {provider} does not support quotes")
        df = prov.fetch_quotes(symbol, start, end)
        if save:
            key = f"{provider}/{key_builder.quotes(symbol)}"
            self._store.write(key, df, mode=mode)
        return df

    def fetch_fundamentals(
        self,
        provider: str,
        symbol: str,
        as_of: date,
        save: bool = False,
    ) -> dict[str, Any]:
        """Fetch fundamentals when supported."""
        prov = self._get_provider(provider)
        if not hasattr(prov, "fetch_fundamentals"):
            raise ValueError(f"Provider {provider} does not support fundamentals")
        data = prov.fetch_fundamentals(symbol, as_of)
        if save:
            key = f"{provider}/{key_builder.fundamentals(symbol)}"
            self._store.write(key, pl.DataFrame([data]), mode="overwrite")
        return data

    def fetch_corporate_actions(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date,
        save: bool = False,
    ) -> list[Any]:
        """Fetch corporate actions when supported."""
        prov = self._get_provider(provider)
        if not hasattr(prov, "get_corporate_actions"):
            raise ValueError(f"Provider {provider} does not support corporate actions")
        actions = prov.get_corporate_actions(symbol, start, end)
        if save and actions:
            key = f"{provider}/{key_builder.corp_actions(symbol)}"
            self._store.write(key, pl.DataFrame(actions), mode="overwrite")
        return actions

    def get_universe(self, provider: str, asset_class: str, as_of: date | None = None) -> list[str]:
        """Return tradeable symbols for an asset class if supported."""
        prov = self._get_provider(provider)
        if not hasattr(prov, "get_universe"):
            raise ValueError(f"Provider {provider} does not support universes")
        return prov.get_universe(asset_class, as_of)

    def fetch_instruments(self, provider: str, asset_class: str) -> list[Any]:
        """Return instruments metadata if supported."""
        prov = self._get_provider(provider)
        if not hasattr(prov, "fetch_instruments"):
            raise ValueError(f"Provider {provider} does not support instrument fetch")
        return prov.fetch_instruments(asset_class)

    def backfill(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date,
        timeframe: str,
    ) -> pl.DataFrame:
        """Fetch only missing ranges based on gaps in stored data."""
        storage_key = self._storage_key(provider, symbol, timeframe)
        existing = self._store.read(storage_key) if self._store.exists(storage_key) else pl.DataFrame()
        prov = self._get_provider(provider)

        if existing.is_empty():
            return self.fetch(provider, symbol, start, end, timeframe=timeframe, save=True, mode="append")

        # If we have too few points to infer gaps, refetch the range
        if existing.height < 2:
            gap_df = prov.fetch_bars(symbol, start, end, timeframe=timeframe)
            if not gap_df.is_empty():
                validate_ohlc(gap_df)
                self._store.write(storage_key, gap_df, mode="append")
                combined = pl.concat([existing, gap_df]).unique(subset=["timestamp"]).sort("timestamp")
                return combined
            return existing

        expected_minutes = _timeframe_to_minutes(timeframe)
        gaps = detect_gaps(existing, timedelta(minutes=expected_minutes))
        fetched_parts: list[pl.DataFrame] = []

        # Cover ends if needed
        if existing["timestamp"].min() > datetime.combine(start, datetime.min.time(), tzinfo=UTC):
            gaps.insert(0, (datetime.combine(start, datetime.min.time(), tzinfo=UTC), existing["timestamp"].min()))
        if existing["timestamp"].max() < datetime.combine(end, datetime.min.time(), tzinfo=UTC):
            gaps.append((existing["timestamp"].max(), datetime.combine(end, datetime.min.time(), tzinfo=UTC)))

        for gap_start, gap_end in gaps:
            gap_df = prov.fetch_bars(symbol, gap_start.date(), gap_end.date(), timeframe=timeframe)
            if not gap_df.is_empty():
                validate_ohlc(gap_df)
                self._store.write(storage_key, gap_df, mode="append")
                fetched_parts.append(gap_df)

        if not fetched_parts:
            return existing

        combined = pl.concat([existing, *fetched_parts]).unique(subset=["timestamp"]).sort("timestamp")
        return combined

    def validate(self, provider: str, symbol: str, timeframe: str) -> dict[str, Any]:
        """Validate data integrity.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            Dict with validation results:
                - valid: Whether data passes validation
                - row_count: Number of rows
                - null_count: Number of null values
                - errors: List of error messages

        Example:
            >>> ds = DataService()
            >>> result = ds.validate("oanda", "EUR_USD", "1m")
            >>> result["valid"]
            True
        """
        df = self.load(provider, symbol, timeframe)

        null_count = sum(df[col].null_count() for col in df.columns)
        dup_count = 0
        if "timestamp" in df.columns:
            dup_count = len(df) - df.select("timestamp").unique().height

        result = {"row_count": len(df), "errors": [], "warnings": [], "null_count": null_count}
        if null_count > 0:
            result["errors"].append(f"Found {null_count} null values")
        if dup_count > 0:
            result["errors"].append(f"Found {dup_count} duplicate timestamps")

        try:
            validation = validate_ohlc(df)
            result["warnings"].extend(validation.warnings)
            result["valid"] = validation.is_valid and not result["errors"]
        except Exception as exc:  # pragma: no cover - already unit tested in qa
            result["valid"] = False
            result["errors"].append(str(exc))
        return result

    def info(self, provider: str, symbol: str, timeframe: str) -> dict[str, Any]:
        """Get data metadata.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            Dict with metadata:
                - row_count: Number of rows
                - columns: List of column names
                - start: First timestamp (date)
                - end: Last timestamp (date)

        Example:
            >>> ds = DataService()
            >>> info = ds.info("oanda", "EUR_USD", "1m")
            >>> info["row_count"]
            1000000
        """
        storage_key = self._storage_key(provider, symbol, timeframe)
        df = self.load(provider, symbol, timeframe)

        result: dict[str, Any] = {
            "row_count": len(df),
            "columns": df.columns,
        }

        # Get timestamp range using store's method
        date_range = self._store.get_date_range(storage_key)
        if date_range:
            result["start"] = date_range[0]
            result["end"] = date_range[1]

        return result

    def stats(self, provider: str, symbol: str, timeframe: str) -> dict[str, Any]:
        """Get statistical summary of data.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            Dict with statistics for each numeric column:
                - mean, min, max, std, count

        Example:
            >>> ds = DataService()
            >>> stats = ds.stats("oanda", "EUR_USD", "1m")
            >>> stats["close"]["mean"]
            1.0875
        """
        df = self.load(provider, symbol, timeframe)

        result: dict[str, Any] = {}
        for col in df.columns:
            if df[col].dtype.is_numeric():
                col_stats = df.select(
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).count().alias("count"),
                )
                result[col] = col_stats.to_dicts()[0]

        return result

    def delete(self, provider: str, symbol: str, timeframe: str) -> bool:
        """Delete data.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            True if data was deleted, False if it didn't exist

        Example:
            >>> ds = DataService()
            >>> ds.delete("oanda", "EUR_USD", "1m")
            True
        """
        storage_key = self._storage_key(provider, symbol, timeframe)
        return self._store.delete(storage_key)

    def exists(self, provider: str, symbol: str, timeframe: str) -> bool:
        """Check if data exists.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe

        Returns:
            True if data exists, False otherwise

        Example:
            >>> ds = DataService()
            >>> ds.exists("oanda", "EUR_USD", "1m")
            True
        """
        storage_key = self._storage_key(provider, symbol, timeframe)
        return self._store.exists(storage_key)

    def gaps(self, provider: str, symbol: str, timeframe: str, expected_minutes: int) -> list[tuple[Any, Any]]:
        """Detect gaps for a symbol/timeframe using expected minutes."""
        df = self.load(provider, symbol, timeframe)
        if df.is_empty():
            return []
        return detect_gaps(df, timedelta(minutes=expected_minutes))

    def validate_credentials(self, provider: str) -> bool:
        """Validate provider credentials via provider adapter."""
        prov = self._get_provider(provider)
        return prov.validate_credentials()


def _timeframe_to_minutes(tf: str) -> int:
    mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    return mapping.get(tf, 1)

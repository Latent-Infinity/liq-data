"""DataService: Programmatic API for liq-data operations.

This module provides a Python API for data operations, wrapping CLI
functionality for programmatic access.

Design Principles:
    - SRP: DataService handles high-level data operations
    - OCP: New operations can be added without modifying existing methods
    - DIP: Depends on abstractions (MarketDataProvider protocol), not implementations
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

import concurrent.futures
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload

import polars as pl
from filelock import FileLock, Timeout

from liq.data.aggregation import aggregate_bars
from liq.data.exceptions import ProviderNoDataError
from liq.data.gaps import detect_gaps
from liq.data.lockbox import USAGE_LOG_FILENAME, LockboxGuard, resolve_dataset
from liq.data.policies import POLICIES
from liq.data.protocols import BatchJob, BatchMarketDataProvider
from liq.data.qa import validate_ohlc
from liq.data.rate_limiter import RateLimiter
from liq.data.settings import (
    LiqDataSettings,
    create_alpaca_provider,
    create_binance_provider,
    create_coinbase_provider,
    create_databento_provider,
    create_fred_provider,
    create_oanda_provider,
    create_polygon_provider,
    create_tradestation_provider,
    get_settings,
)
from liq.data.sync_events import (
    EVENT_MANIFEST_GAP_DETECTED,
    EVENT_MANIFEST_RANGE_APPENDED,
    EVENT_MANIFEST_ROLLBACK,
    EVENT_PIT_WARNING,
    EVENT_SYMBOL_COMPLETED,
    EVENT_SYMBOL_FAILED,
    EVENT_SYMBOL_STARTED,
    EVENT_SYNC_COMPLETED,
    EVENT_SYNC_STARTED,
    EVENT_UNIVERSE_RESOLVED,
    SyncLockedError,
)
from liq.store import key_builder
from liq.store.parquet import ParquetStore

if TYPE_CHECKING:
    from liq.data.protocols import MarketDataProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _BatchSyncWork:
    symbol: str
    start: datetime
    end: datetime


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
    _PROVIDER_FACTORIES: ClassVar[dict[str, Callable[[LiqDataSettings], MarketDataProvider]]] = {
        "oanda": create_oanda_provider,
        "binance": create_binance_provider,
        "tradestation": create_tradestation_provider,
        "coinbase": create_coinbase_provider,
        "polygon": create_polygon_provider,
        "alpaca": create_alpaca_provider,
        "fred": create_fred_provider,
        "databento": create_databento_provider,
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
        self._lockbox_guard: LockboxGuard | None = None

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

    @property
    def lockbox_guard(self) -> LockboxGuard:
        """Lockbox guard for research reads (usage log lives under data_root)."""
        if self._lockbox_guard is None:
            self._lockbox_guard = LockboxGuard(usage_log_path=self._data_root / USAGE_LOG_FILENAME)
        return self._lockbox_guard

    def _guard_research_read(
        self,
        provider: str,
        symbol: str,
        start: date | None,
        end: date | None,
        *,
        purpose: str | None,
        arm_id: str | None,
        final_portfolio_review: bool,
    ) -> None:
        """Route a declared-purpose read through the lockbox guard."""
        if purpose is None:
            return
        if not arm_id:
            raise ValueError("arm_id is required when a research purpose is declared")
        dataset = resolve_dataset(provider, symbol)
        if dataset is None:
            return
        self.lockbox_guard.assert_period_allowed(
            dataset,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
        )

    def _storage_key(self, provider: str, symbol: str, timeframe: str) -> str:
        """Build storage key for bars via provider/key_builder."""
        return f"{provider}/{key_builder.bars(symbol, timeframe)}"

    def _get_provider(self, provider_name: str) -> MarketDataProvider:
        """Get a provider instance by name.

        Args:
            provider_name: Provider name (e.g., "oanda", "binance")

        Returns:
            MarketDataProvider instance

        Raises:
            ValueError: If provider is not supported
        """
        factory = self._PROVIDER_FACTORIES.get(provider_name.lower())
        if factory is None:
            supported = ", ".join(self._PROVIDER_FACTORIES.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Supported: {supported}")
        provider = factory(self._settings)
        set_store = getattr(provider, "set_store", None)
        if callable(set_store):
            set_store(self._store)
        return provider

    @overload
    def load(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date | None = ...,
        end: date | None = ...,
        columns: list[str] | None = ...,
        *,
        streaming: Literal[False] = False,
        batch_size: None = None,
        purpose: str | None = ...,
        arm_id: str | None = ...,
        final_portfolio_review: bool = ...,
    ) -> pl.DataFrame: ...

    @overload
    def load(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date | None = ...,
        end: date | None = ...,
        columns: list[str] | None = ...,
        *,
        streaming: bool = ...,
        batch_size: int | None = ...,
        purpose: str | None = ...,
        arm_id: str | None = ...,
        final_portfolio_review: bool = ...,
    ) -> pl.DataFrame | Iterator[pl.DataFrame]: ...

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
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
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
            purpose: Research purpose of the read (e.g., "discovery",
                "validation", "dev_smoke"). Declaring a purpose routes the
                read through the lockbox guard; research reads MUST declare
                one. Reads without a purpose are never research evidence.
            arm_id: Research arm consuming the read (required with purpose)
            final_portfolio_review: Human-only flag permitting program-lockbox
                reads; recorded in the usage log

        Returns:
            DataFrame with OHLCV data

        Raises:
            FileNotFoundError: If data doesn't exist
            LockboxViolationError: If a declared-purpose read violates the
                lockbox ledger fold boundaries

        Example:
            >>> ds = DataService()
            >>> df = ds.load("oanda", "EUR_USD", "1m")
        """
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
        )
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
                        raise ValueError("batch_size is not supported for aggregated reads")
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
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
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
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
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
                result.append(
                    {
                        "provider": parts[0],
                        "symbol": parts[1],
                        "timeframe": parts[3],
                    }
                )

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
        existing = (
            self._store.read(storage_key) if self._store.exists(storage_key) else pl.DataFrame()
        )
        prov = self._get_provider(provider)

        if existing.is_empty():
            return self.fetch(
                provider, symbol, start, end, timeframe=timeframe, save=True, mode="append"
            )

        # If we have too few points to infer gaps, refetch the range
        if existing.height < 2:
            gap_df = prov.fetch_bars(symbol, start, end, timeframe=timeframe)
            if not gap_df.is_empty():
                validate_ohlc(gap_df)
                self._store.write(storage_key, gap_df, mode="append")
                combined = (
                    pl.concat([existing, gap_df]).unique(subset=["timestamp"]).sort("timestamp")
                )
                return combined
            return existing

        expected_minutes = _timeframe_to_minutes(timeframe)
        gaps = detect_gaps(existing, timedelta(minutes=expected_minutes))
        fetched_parts: list[pl.DataFrame] = []

        # Cover ends if needed.
        # Polars typing widens Series.min()/.max() to PythonLiteral | None;
        # we've already validated the column is a tz-aware datetime above,
        # so narrow back to datetime for the comparisons + tuple append.
        existing_min = cast(datetime, existing["timestamp"].min())
        existing_max = cast(datetime, existing["timestamp"].max())
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(end, datetime.min.time(), tzinfo=UTC)
        if existing_min > start_dt:
            gaps.insert(0, (start_dt, existing_min))
        if existing_max < end_dt:
            gaps.append((existing_max, end_dt))

        for gap_start, gap_end in gaps:
            gap_df = prov.fetch_bars(symbol, gap_start.date(), gap_end.date(), timeframe=timeframe)
            if not gap_df.is_empty():
                validate_ohlc(gap_df)
                self._store.write(storage_key, gap_df, mode="append")
                fetched_parts.append(gap_df)

        if not fetched_parts:
            return existing

        combined = (
            pl.concat([existing, *fetched_parts]).unique(subset=["timestamp"]).sort("timestamp")
        )
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

        errors: list[str] = []
        warnings: list[str] = []
        result: dict[str, Any] = {
            "row_count": len(df),
            "errors": errors,
            "warnings": warnings,
            "null_count": null_count,
        }
        if null_count > 0:
            errors.append(f"Found {null_count} null values")
        if dup_count > 0:
            errors.append(f"Found {dup_count} duplicate timestamps")

        try:
            validation = validate_ohlc(df)
            warnings.extend(validation.warnings)
            result["valid"] = validation.is_valid and not errors
        except Exception as exc:  # pragma: no cover - already unit tested in qa
            result["valid"] = False
            errors.append(str(exc))
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

    def gaps(
        self, provider: str, symbol: str, timeframe: str, expected_minutes: int
    ) -> list[tuple[Any, Any]]:
        """Detect gaps for a symbol/timeframe using expected minutes."""
        df = self.load(provider, symbol, timeframe)
        if df.is_empty():
            return []
        return detect_gaps(df, timedelta(minutes=expected_minutes))

    def validate_credentials(self, provider: str) -> bool:
        """Validate provider credentials via provider adapter."""
        prov = self._get_provider(provider)
        return prov.validate_credentials()

    # ----- universe resolution / sync ----------------------------------

    def resolve_universe(
        self,
        universe_ref: Any,
        *,
        as_of: date,
        registry: Any | None = None,
    ) -> Any:
        """Resolve a universe reference as-of a date.

        Accepts three input shapes for caller convenience:

        * :class:`UniverseDefinition`: used verbatim.
        * ``list[str]`` of symbols: wrapped in an explicit definition.
        * ``str`` name: looked up in ``registry``; raises ``ValueError``
          if no registry is supplied.

        Returns a :class:`ResolvedUniverse` whose ``symbols`` are
        normalized (uppercase, deduped, sorted) and whose ``pit`` flag
        carries the constituent-source semantics.
        """
        from liq.data.universes import (
            InMemoryStubSource,
            UniverseDefinition,
            UniverseKind,
            UniverseRegistry,
            UniverseResolver,
        )

        if isinstance(universe_ref, UniverseDefinition):
            definition = universe_ref
        elif isinstance(universe_ref, list):
            definition = UniverseDefinition(
                name="adhoc",
                version=1,
                kind=UniverseKind.EXPLICIT,
                spec={"symbols": [str(s) for s in universe_ref]},
            )
        else:
            reg = registry if isinstance(registry, UniverseRegistry) else None
            if reg is None:
                raise ValueError(
                    "resolve_universe() with a universe name requires a UniverseRegistry"
                )
            definition = reg.load(str(universe_ref))

        named_universes = None
        if isinstance(registry, UniverseRegistry):
            named_universes = {n: registry.load(n) for n in registry.list_names()}

        return UniverseResolver(
            constituent_source=InMemoryStubSource(),
            named_universes=named_universes,
        ).resolve(definition, as_of=as_of)

    def estimate_databento_cost(
        self,
        universe: Any,
        *,
        start: date,
        end: date,
        timeframe: str,
        dataset: str,
        registry: Any | None = None,
    ) -> dict[str, Any]:
        """No-spend cost estimate for a Databento ingestion.

        Wraps Databento's non-billable metadata endpoints
        (``metadata.get_cost`` and ``metadata.get_billable_size``) so a
        caller can confirm an operator-authorised cost bound *before*
        invoking the billable :meth:`sync`. Currently only
        ``timeframe="1m"`` is supported (derives ``schema="ohlcv-1m"``).

        Returns a dict with keys:
            billable_bytes, estimated_cost_usd, dataset, schema,
            symbols, start, end, provider_request_id

        Raises ``ValueError`` on bad timeframe, empty universe, inverted
        date range, or missing ``DATABENTO_API_KEY``.
        """
        if timeframe != "1m":
            raise ValueError(
                f"estimate_databento_cost only supports timeframe='1m' "
                f"(got {timeframe!r}); higher timeframes are not yet wired."
            )
        if end < start:
            raise ValueError(f"end ({end}) must be on or after start ({start})")
        if not self._settings.databento_api_key:
            raise ValueError(
                "DATABENTO_API_KEY not configured. Set it in .env or pass "
                "settings with `databento_api_key=...`."
            )

        resolved = self.resolve_universe(universe, as_of=end, registry=registry)
        symbols = list(resolved.symbols)
        if not symbols:
            raise ValueError(
                "estimate_databento_cost requires a non-empty symbols list "
                "after universe resolution"
            )

        schema = "ohlcv-1m"

        # Lazy import keeps the databento dep cost off the hot path.
        import databento  # noqa: PLC0415

        client = databento.Historical(key=self._settings.databento_api_key)
        cost_usd = float(
            client.metadata.get_cost(
                dataset=dataset,
                symbols=symbols,
                schema=schema,
                start=start,
                end=end,
            )
        )
        billable_bytes = int(
            client.metadata.get_billable_size(
                dataset=dataset,
                symbols=symbols,
                schema=schema,
                start=start,
                end=end,
            )
        )

        request_id = uuid.uuid4().hex
        logger.info(
            "databento cost estimate",
            extra={
                "provider": "databento",
                "dataset": dataset,
                "schema": schema,
                "symbols": symbols,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "estimated_cost_usd": cost_usd,
                "billable_bytes": billable_bytes,
                "provider_request_id": request_id,
            },
        )

        return {
            "billable_bytes": billable_bytes,
            "estimated_cost_usd": cost_usd,
            "dataset": dataset,
            "schema": schema,
            "symbols": symbols,
            "start": start,
            "end": end,
            "provider_request_id": request_id,
        }

    def sync(
        self,
        universe: Any,
        *,
        start: date,
        end: date,
        provider: str,
        timeframe: str,
        dataset: str,
        force_refresh: bool = False,
        registry: Any | None = None,
        as_of: date | None = None,
        lock_timeout: float = 60.0,
        max_workers: int = 1,
    ) -> dict[str, Any]:
        """Sync a universe to local storage.

        Resolves the universe (a :class:`UniverseDefinition` or a name to
        look up in ``registry``), computes per-symbol fetch gaps from the
        coverage manifest, fetches each gap via the provider, and
        records the new ranges transactionally per symbol, so a fetch
        that raises mid-flight rolls back the manifest claim for that
        symbol but leaves earlier-symbol claims persisted (resumable).

        A file lock keyed on ``(provider, dataset, timeframe,
        universe.name)`` serializes concurrent syncs against the same
        target; ``lock_timeout`` controls how long to wait before
        raising :class:`SyncLockedError`.

        ``max_workers`` (default ``1``: sequential, byte-identical to
        the pre-enhancement behaviour) caps the size of the
        :class:`concurrent.futures.ThreadPoolExecutor` used for the
        per-symbol loop. Values > 1 unlock per-symbol concurrency
        within the universe-level file lock; per-symbol transactional
        manifest claims are preserved unchanged (the lock around each
        symbol's manifest transaction is the inherited ``txn`` context
        manager).

        Returns a small report dict:

        * ``symbols``: count of resolved symbols
        * ``api_calls``: number of provider fetches actually made
          (zero on a fully-cached re-run)
        * ``manifest_gaps``: total uncovered ranges seen at planning time
        * ``rows_fetched``: total rows pulled across all symbols
        * ``pit``: propagated from the resolved universe
        * ``sync_run_id``: UUID correlating every emitted log event

        ``force_refresh=True`` skips the gap calc and re-fetches every
        symbol over the full requested window.
        """
        from liq.data.manifest import CoverageManifest, CoverageRange
        from liq.data.universes import (
            InMemoryStubSource,
            UniverseDefinition,
            UniverseRegistry,
            UniverseResolver,
        )

        if isinstance(universe, UniverseDefinition):
            definition = universe
        else:
            reg = registry if isinstance(registry, UniverseRegistry) else None
            if reg is None:
                raise ValueError("sync() with a universe name requires a UniverseRegistry")
            definition = reg.load(str(universe))
        named_universes = None
        if isinstance(registry, UniverseRegistry):
            named_universes = {name: registry.load(name) for name in registry.list_names()}

        resolved = UniverseResolver(
            constituent_source=InMemoryStubSource(),
            named_universes=named_universes,
        ).resolve(definition, as_of=as_of or end)

        sync_run_id = uuid.uuid4().hex
        log_base: dict[str, Any] = {
            "sync_run_id": sync_run_id,
            "universe": definition.name,
            "version": definition.version,
            "kind": definition.kind.value,
            "provider": provider,
            "dataset": dataset,
            "timeframe": timeframe,
        }
        logger.info(
            "sync started",
            extra={
                **log_base,
                "event": EVENT_SYNC_STARTED,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "force_refresh": force_refresh,
            },
        )
        logger.info(
            "universe resolved",
            extra={
                **log_base,
                "event": EVENT_UNIVERSE_RESOLVED,
                "symbols_count": len(resolved.symbols),
                "as_of": resolved.as_of.isoformat(),
                "pit": resolved.pit,
            },
        )
        if not resolved.pit:
            logger.warning(
                "non-PIT universe; downstream sweeps will reject",
                extra={
                    **log_base,
                    "event": EVENT_PIT_WARNING,
                    "reason": "constituent source did not advertise PIT membership",
                },
            )

        lock_path = (
            self._data_root
            / "locks"
            / "sync"
            / f"{provider}--{dataset}--{timeframe}--{definition.name}.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path), timeout=lock_timeout)
        lock_acquired = False
        try:
            lock.acquire()
            lock_acquired = True
        except Timeout as exc:
            raise SyncLockedError(
                f"sync lock held for {provider}/{dataset}/{timeframe}/{definition.name}; "
                f"waited {lock_timeout}s"
            ) from exc

        try:
            prov = self._get_provider(provider)
            policy = POLICIES.get(provider)
            rate_limiter = RateLimiter(
                requests_per_minute=policy.requests_per_minute if policy else None,
                burst=policy.burst if policy else None,
                min_interval_seconds=policy.min_interval_seconds if policy else None,
            )
            fetched_at = datetime.now(UTC)
            api_calls = 0
            manifest_gaps_total = 0
            rows_fetched = 0

            request_start = datetime(start.year, start.month, start.day, tzinfo=UTC)
            request_end = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)

            if max_workers <= 0:
                raise ValueError(f"max_workers must be >= 1; got {max_workers}")

            counters_lock = threading.Lock()
            rate_limiter_lock = threading.Lock()

            def _sync_one_symbol(symbol: str) -> None:
                """Per-symbol fetch + manifest-transaction body.

                Identical semantics to the sequential implementation; lifted
                out so the outer loop can dispatch it either directly
                (max_workers=1, byte-identical to the prior code path) or via
                a ThreadPoolExecutor (max_workers > 1).
                """
                nonlocal api_calls, rows_fetched, manifest_gaps_total

                manifest = CoverageManifest.load(
                    root=self._data_root,
                    provider=provider,
                    dataset=dataset,
                    timeframe=timeframe,
                    symbol=symbol,
                )
                gaps = (
                    [(request_start, request_end)]
                    if force_refresh
                    else manifest.gaps(start=request_start, end=request_end)
                )
                with counters_lock:
                    manifest_gaps_total += len(gaps)
                if not gaps:
                    return

                logger.info(
                    "manifest gaps detected",
                    extra={
                        **log_base,
                        "event": EVENT_MANIFEST_GAP_DETECTED,
                        "symbol": symbol,
                        "gaps_count": len(gaps),
                    },
                )
                logger.info(
                    "symbol started",
                    extra={
                        **log_base,
                        "event": EVENT_SYMBOL_STARTED,
                        "symbol": symbol,
                        "gaps_count": len(gaps),
                    },
                )
                symbol_rows = 0

                try:
                    with manifest.transaction() as txn:
                        for gap_start, gap_end in gaps:
                            with rate_limiter_lock:
                                rate_limiter.acquire()
                            df = prov.fetch_bars(
                                symbol,
                                gap_start.date(),
                                gap_end.date(),
                                timeframe=timeframe,
                            )
                            with counters_lock:
                                api_calls += 1
                                rows_fetched += df.height
                            symbol_rows += df.height
                            if not df.is_empty():
                                storage_key = self._storage_key(provider, symbol, timeframe)
                                write_mode = "overwrite" if force_refresh else "append"
                                self._store.write(storage_key, df, mode=write_mode)
                            txn.record(
                                CoverageRange(
                                    start=gap_start,
                                    end=gap_end,
                                    fetched_at=fetched_at,
                                )
                            )
                            logger.info(
                                "manifest range appended",
                                extra={
                                    **log_base,
                                    "event": EVENT_MANIFEST_RANGE_APPENDED,
                                    "symbol": symbol,
                                    "start": gap_start.isoformat(),
                                    "end": gap_end.isoformat(),
                                    "rows": df.height,
                                },
                            )
                except Exception as exc:
                    logger.error(
                        "manifest rollback",
                        extra={
                            **log_base,
                            "event": EVENT_MANIFEST_ROLLBACK,
                            "symbol": symbol,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    logger.error(
                        "symbol failed",
                        extra={
                            **log_base,
                            "event": EVENT_SYMBOL_FAILED,
                            "symbol": symbol,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        },
                    )
                    raise
                logger.info(
                    "symbol completed",
                    extra={
                        **log_base,
                        "event": EVENT_SYMBOL_COMPLETED,
                        "symbol": symbol,
                        "rows": symbol_rows,
                        "gaps_count": len(gaps),
                    },
                )

            if max_workers == 1:
                # Sequential dispatch: byte-identical to the prior code path
                # (no thread-pool overhead, no spurious ordering changes).
                for symbol in resolved.symbols:
                    _sync_one_symbol(symbol)
            else:
                # Parallel dispatch: bounded to ``max_workers`` concurrent
                # in-flight per-symbol tasks. The per-symbol manifest
                # transaction is the inherited atomic claim. The first
                # exception aborts the sync (waiting on in-flight tasks to
                # drain naturally via ThreadPoolExecutor shutdown).
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="liq-data-sync",
                ) as pool:
                    futures = [pool.submit(_sync_one_symbol, symbol) for symbol in resolved.symbols]
                    for future in concurrent.futures.as_completed(futures):
                        # Re-raise the first exception we observe; outstanding
                        # futures continue running to their own completion
                        # under the executor's shutdown semantics.
                        future.result()

            logger.info(
                "sync completed",
                extra={
                    **log_base,
                    "event": EVENT_SYNC_COMPLETED,
                    "symbols": len(resolved.symbols),
                    "api_calls": api_calls,
                    "rows_fetched": rows_fetched,
                    "manifest_gaps": manifest_gaps_total,
                },
            )

            return {
                "symbols": len(resolved.symbols),
                "api_calls": api_calls,
                "manifest_gaps": manifest_gaps_total,
                "rows_fetched": rows_fetched,
                "force_refresh": force_refresh,
                "pit": resolved.pit,
                "sync_run_id": sync_run_id,
            }
        finally:
            if lock_acquired:
                lock.release()

    def sync_batch(
        self,
        universe: Any,
        *,
        start: date,
        end: date,
        provider: str,
        timeframe: str,
        dataset: str,
        force_refresh: bool = False,
        registry: Any | None = None,
        as_of: date | None = None,
        max_in_flight: int = 4,
        lock_timeout: float = 60.0,
        poll_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Sync a universe using a provider's external batch lifecycle.

        Providers opt in via :class:`BatchMarketDataProvider`: submit a
        durable batch job, poll until it is ready, then download/materialize
        it. The service owns orchestration, manifest transactions, storage
        writes, and progress events, so the same recovery semantics as
        :meth:`sync` apply after each completed batch is downloaded.
        """
        from liq.data.manifest import CoverageManifest, CoverageRange
        from liq.data.universes import (
            InMemoryStubSource,
            UniverseDefinition,
            UniverseRegistry,
            UniverseResolver,
        )

        if max_in_flight < 1:
            raise ValueError(f"max_in_flight must be >= 1, got {max_in_flight}")

        if isinstance(universe, UniverseDefinition):
            definition = universe
        else:
            reg = registry if isinstance(registry, UniverseRegistry) else None
            if reg is None:
                raise ValueError("sync_batch() with a universe name requires a UniverseRegistry")
            definition = reg.load(str(universe))
        named_universes = None
        if isinstance(registry, UniverseRegistry):
            named_universes = {name: registry.load(name) for name in registry.list_names()}

        resolved = UniverseResolver(
            constituent_source=InMemoryStubSource(),
            named_universes=named_universes,
        ).resolve(definition, as_of=as_of or end)

        sync_run_id = uuid.uuid4().hex
        log_base: dict[str, Any] = {
            "sync_run_id": sync_run_id,
            "universe": definition.name,
            "version": definition.version,
            "kind": definition.kind.value,
            "provider": provider,
            "dataset": dataset,
            "timeframe": timeframe,
        }
        logger.info(
            "sync started",
            extra={
                **log_base,
                "event": EVENT_SYNC_STARTED,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "force_refresh": force_refresh,
                "orchestration": "batch",
                "max_in_flight": max_in_flight,
            },
        )
        logger.info(
            "universe resolved",
            extra={
                **log_base,
                "event": EVENT_UNIVERSE_RESOLVED,
                "symbols_count": len(resolved.symbols),
                "as_of": resolved.as_of.isoformat(),
                "pit": resolved.pit,
            },
        )
        if not resolved.pit:
            logger.warning(
                "non-PIT universe; downstream sweeps will reject",
                extra={
                    **log_base,
                    "event": EVENT_PIT_WARNING,
                    "reason": "constituent source did not advertise PIT membership",
                },
            )

        lock_path = (
            self._data_root
            / "locks"
            / "sync"
            / f"{provider}--{dataset}--{timeframe}--{definition.name}.lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(lock_path), timeout=lock_timeout)
        lock_acquired = False
        try:
            lock.acquire()
            lock_acquired = True
        except Timeout as exc:
            raise SyncLockedError(
                f"sync lock held for {provider}/{dataset}/{timeframe}/{definition.name}; "
                f"waited {lock_timeout}s"
            ) from exc

        try:
            prov = self._get_provider(provider)
            if not all(
                callable(getattr(prov, name, None))
                for name in (
                    "submit_batch_bars",
                    "poll_batch_bars",
                    "fetch_completed_batch_bars",
                )
            ):
                raise ValueError(f"Provider {provider!r} does not support batch orchestration")
            batch_provider = cast(BatchMarketDataProvider, prov)
            policy = POLICIES.get(provider)
            rate_limiter = RateLimiter(
                requests_per_minute=policy.requests_per_minute if policy else None,
                burst=policy.burst if policy else None,
                min_interval_seconds=policy.min_interval_seconds if policy else None,
            )
            fetched_at = datetime.now(UTC)
            api_calls = 0
            manifest_gaps_total = 0
            rows_fetched = 0
            symbols_skipped = 0
            skipped_symbols: list[str] = []
            request_start = datetime(start.year, start.month, start.day, tzinfo=UTC)
            request_end = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)

            pending: list[_BatchSyncWork] = []
            symbol_rows: dict[str, int] = {}
            symbol_remaining: dict[str, int] = {}
            symbol_gap_counts: dict[str, int] = {}
            for symbol in resolved.symbols:
                manifest = CoverageManifest.load(
                    root=self._data_root,
                    provider=provider,
                    dataset=dataset,
                    timeframe=timeframe,
                    symbol=symbol,
                )
                gaps = (
                    [(request_start, request_end)]
                    if force_refresh
                    else manifest.gaps(start=request_start, end=request_end)
                )
                manifest_gaps_total += len(gaps)
                if not gaps:
                    continue
                logger.info(
                    "manifest gaps detected",
                    extra={
                        **log_base,
                        "event": EVENT_MANIFEST_GAP_DETECTED,
                        "symbol": symbol,
                        "gaps_count": len(gaps),
                    },
                )
                logger.info(
                    "symbol started",
                    extra={
                        **log_base,
                        "event": EVENT_SYMBOL_STARTED,
                        "symbol": symbol,
                        "gaps_count": len(gaps),
                    },
                )
                symbol_rows[symbol] = 0
                symbol_remaining[symbol] = len(gaps)
                symbol_gap_counts[symbol] = len(gaps)
                pending.extend(_BatchSyncWork(symbol=symbol, start=a, end=b) for a, b in gaps)

            active: dict[str, tuple[BatchJob, _BatchSyncWork]] = {}
            pause = (
                poll_seconds
                if poll_seconds is not None
                else float(getattr(batch_provider, "batch_poll_seconds", 5.0))
            )
            while pending or active:
                while pending and len(active) < max_in_flight:
                    work = pending.pop(0)
                    rate_limiter.acquire()
                    try:
                        job = batch_provider.submit_batch_bars(
                            work.symbol,
                            work.start.date(),
                            work.end.date(),
                            timeframe,
                            dataset=dataset,
                            sync_run_id=sync_run_id,
                        )
                    except Exception as exc:
                        # Surface the failing symbol before propagating so
                        # log readers can correlate the abort with one
                        # row of the universe rather than discovering it
                        # via stacktrace alone.
                        logger.error(
                            "symbol failed",
                            extra={
                                **log_base,
                                "event": EVENT_SYMBOL_FAILED,
                                "symbol": work.symbol,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "stage": "submit",
                            },
                        )
                        if isinstance(exc, ProviderNoDataError):
                            symbols_skipped += 1
                            skipped_symbols.append(work.symbol)
                            symbol_remaining[work.symbol] -= 1
                            if symbol_remaining[work.symbol] == 0:
                                logger.info(
                                    "symbol completed",
                                    extra={
                                        **log_base,
                                        "event": EVENT_SYMBOL_COMPLETED,
                                        "symbol": work.symbol,
                                        "rows": 0,
                                        "gaps_count": symbol_gap_counts[work.symbol],
                                        "skipped": True,
                                    },
                                )
                            continue
                        raise
                    active[job.signature] = (job, work)

                # Polling is a status check, not a billable data request.
                # If we charged it to the rate limiter, a high
                # ``max_in_flight`` value would burn through the per-
                # minute budget on poll ticks alone and starve the
                # actual submits + downloads. The provider's
                # ``poll_batch_bars`` already retries transient SDK
                # failures internally.
                completed: list[str] = []
                for signature, (job, work) in list(active.items()):
                    try:
                        ready = batch_provider.poll_batch_bars(job, sync_run_id=sync_run_id)
                    except Exception as exc:
                        logger.error(
                            "symbol failed",
                            extra={
                                **log_base,
                                "event": EVENT_SYMBOL_FAILED,
                                "symbol": work.symbol,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "stage": "poll",
                            },
                        )
                        raise
                    if ready:
                        completed.append(signature)

                if not completed:
                    if active and pause > 0:
                        time.sleep(pause)
                    continue

                for signature in completed:
                    job, work = active.pop(signature)
                    manifest = CoverageManifest.load(
                        root=self._data_root,
                        provider=provider,
                        dataset=dataset,
                        timeframe=timeframe,
                        symbol=work.symbol,
                    )
                    try:
                        with manifest.transaction() as txn:
                            rate_limiter.acquire()
                            df = batch_provider.fetch_completed_batch_bars(
                                job, sync_run_id=sync_run_id
                            )
                            api_calls += 1
                            rows_fetched += df.height
                            symbol_rows[work.symbol] += df.height
                            if not df.is_empty():
                                storage_key = self._storage_key(provider, work.symbol, timeframe)
                                write_mode = "overwrite" if force_refresh else "append"
                                self._store.write(storage_key, df, mode=write_mode)
                            txn.record(
                                CoverageRange(
                                    start=work.start,
                                    end=work.end,
                                    fetched_at=fetched_at,
                                )
                            )
                            logger.info(
                                "manifest range appended",
                                extra={
                                    **log_base,
                                    "event": EVENT_MANIFEST_RANGE_APPENDED,
                                    "symbol": work.symbol,
                                    "start": work.start.isoformat(),
                                    "end": work.end.isoformat(),
                                    "rows": df.height,
                                },
                            )
                    except Exception as exc:
                        logger.error(
                            "manifest rollback",
                            extra={
                                **log_base,
                                "event": EVENT_MANIFEST_ROLLBACK,
                                "symbol": work.symbol,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            },
                        )
                        logger.error(
                            "symbol failed",
                            extra={
                                **log_base,
                                "event": EVENT_SYMBOL_FAILED,
                                "symbol": work.symbol,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            },
                        )
                        raise
                    symbol_remaining[work.symbol] -= 1
                    if symbol_remaining[work.symbol] == 0:
                        logger.info(
                            "symbol completed",
                            extra={
                                **log_base,
                                "event": EVENT_SYMBOL_COMPLETED,
                                "symbol": work.symbol,
                                "rows": symbol_rows[work.symbol],
                                "gaps_count": symbol_gap_counts[work.symbol],
                            },
                        )

            logger.info(
                "sync completed",
                extra={
                    **log_base,
                    "event": EVENT_SYNC_COMPLETED,
                    "symbols": len(resolved.symbols),
                    "api_calls": api_calls,
                    "rows_fetched": rows_fetched,
                    "manifest_gaps": manifest_gaps_total,
                    "symbols_skipped": symbols_skipped,
                },
            )
            return {
                "symbols": len(resolved.symbols),
                "api_calls": api_calls,
                "manifest_gaps": manifest_gaps_total,
                "rows_fetched": rows_fetched,
                "force_refresh": force_refresh,
                "pit": resolved.pit,
                "sync_run_id": sync_run_id,
                "orchestration": "batch",
                "max_in_flight": max_in_flight,
                "symbols_skipped": symbols_skipped,
                "skipped_symbols": skipped_symbols,
            }
        finally:
            if lock_acquired:
                lock.release()


def _timeframe_to_minutes(tf: str) -> int:
    mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    return mapping.get(tf, 1)

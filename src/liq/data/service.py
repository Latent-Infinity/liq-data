"""DataService: Programmatic API for liq-data operations.

This module provides a Python API for data operations, wrapping CLI
functionality for programmatic access.

Design Principles:
    - SRP: DataService handles high-level data operations
    - OCP: New operations can be added without modifying existing methods
    - DIP: Depends on abstractions (MarketDataProvider protocol), not implementations
    - DRY: Uses liq-store for all storage operations (single source of truth)

Declaring reads:
    Every entry point that can return market values is default-deny. A read
    must declare what it is for -- ``purpose`` plus ``arm_id`` -- or it is
    refused with :class:`~liq.data.exceptions.UndeclaredReadError` before any
    store or provider is touched. Research reads declare one of
    ``discovery`` / ``validation`` / ``characterization`` / ``forward_accrual``
    and are checked against the lockbox ledger's fold windows. Reads that are
    not research -- ingestion, fixture builds, diagnostics -- declare
    ``dev_smoke`` and name the component in the free-form ``arm_id``;
    ``dev_smoke`` reads are logged and are never research evidence.

Example:
    from liq.data import DataService
    from datetime import date

    # Initialize service
    ds = DataService()

    # Fetch data from provider
    df = ds.fetch(
        "oanda", "EUR_USD", date(2024, 1, 1), date(2024, 1, 31),
        purpose="dev_smoke", arm_id="my_component",
    )

    # Load existing data
    df = ds.load(
        "oanda", "EUR_USD", "1m", date(2024, 1, 1), date(2024, 1, 31),
        purpose="discovery", arm_id="idea_05a",
    )

    # List available data (metadata only; no declaration needed)
    available = ds.list_symbols()

    # Validate data integrity
    result = ds.validate(
        "oanda", "EUR_USD", "1m", purpose="dev_smoke", arm_id="my_component"
    )
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload

import polars as pl
from filelock import FileLock, Timeout

from liq.data.aggregation import aggregate_bars
from liq.data.alpaca_announcement_source import (
    AlpacaAnnouncementCalendarRequest,
    AlpacaAnnouncementDailyRequest,
    validate_announcement_request,
)
from liq.data.alpaca_event_basis_source import (
    AlpacaEventBasisDailyRequest,
    validate_event_basis_request,
)
from liq.data.alpaca_momentum_source import (
    AlpacaMomentumCalendarRequest,
    AlpacaMomentumDailyRequest,
)
from liq.data.alpaca_source import (
    AlpacaBarsRequest,
    AlpacaCalendarRequest,
    AlpacaCorporateActionsRequest,
    AlpacaDailyFixtureRequest,
)
from liq.data.exceptions import (
    LockboxViolationError,
    ProviderError,
    ProviderNoDataError,
    SchemaValidationError,
    StorageVenueMismatchError,
    StoredBarsRequiredError,
    TimestampAuditLimitError,
    UnclassifiedProviderError,
    UndeclaredReadError,
    ValidationError,
)
from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.gaps import detect_gaps
from liq.data.lockbox import (
    CLASSICAL_RANKER_ALPACA_LEDGER_V1,
    CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1,
    CLASSICAL_RANKER_CALENDAR_REFERENCE_LEDGER_V1,
    CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1,
    CLASSICAL_RANKER_EVENT_BASIS_FIXTURE_LEDGER_V1,
    CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1,
    RESEARCH_PURPOSES,
    UNGOVERNED_PROVIDERS,
    USAGE_LOG_FILENAME,
    LockboxGuard,
    LockboxLedger,
    declared_storage_venue,
)
from liq.data.oanda_instrument_metadata import (
    InstrumentMetadataObservation,
    InstrumentMetadataRequest,
)
from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.oanda_source_metadata import CandleDay, CandleDayObservation
from liq.data.policies import POLICIES
from liq.data.protocols import (
    AccountCandleProbeProvider,
    AccountSourceDayProvider,
    BatchJob,
    BatchMarketDataProvider,
    BidAskBarProvider,
    CorporateActionsProvider,
    SourcePriceCaptureProvider,
)
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.providers.databento import SCHEMA_BY_TIMEFRAME
from liq.data.providers.oanda_account_probe import PROBE_ARM, PROBE_START, AccountCandleObservation
from liq.data.providers.oanda_price_capture import PriceCaptureFailure, PriceCapturePolicyRefused
from liq.data.providers.oanda_source_body import SourceBody
from liq.data.providers.oanda_source_transport import validate_source_day
from liq.data.qa import validate_ohlc
from liq.data.rate_limiter import RateLimiter
from liq.data.security_master import (
    SecurityMasterProvider,
    SecurityMasterRecord,
    validate_security_master_request,
)
from liq.data.settings import (
    LiqDataSettings,
    create_alpaca_provider,
    create_binance_provider,
    create_coinbase_provider,
    create_databento_provider,
    create_fred_provider,
    create_oanda_provider,
    create_polygon_provider,
    create_sec_edgar_provider,
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
from liq.data.timestamp_domain import (
    TimestampDomainAccumulator,
    TimestampDomainAudit,
    TimestampDomainRefused,
)
from liq.store import key_builder
from liq.store.parquet import ParquetStore
from liq.store.timestamp_metadata import (
    PinnedParquetFile,
    TimestampFileMetadata,
    TimestampMetadataRefused,
    inspect_timestamp_metadata,
)
from liq.store.timestamp_projection import TimestampProjectionLimits, iter_pinned_timestamps

if TYPE_CHECKING:
    from liq.data.protocols import MarketDataProvider

logger = logging.getLogger(__name__)


def _require_declared_purpose(purpose: str | None, entry_point: str) -> str:
    """Refuse a read that declares no ``purpose``, before anything is read.

    The guard is default-deny: what a read is for is not optional information.
    The raised type is deliberately distinct from
    :class:`~liq.data.exceptions.LockboxViolationError`, because "you did not
    say" and "you may not" are different facts.

    Returns the declared purpose so callers carry a non-optional value onward.
    """
    if purpose is None:
        raise UndeclaredReadError(
            f"{entry_point} declares no purpose and is refused: pass purpose= and "
            f"arm_id=. Research reads declare one of {sorted(RESEARCH_PURPOSES)}; "
            "reads that are not research declare 'dev_smoke' and name the "
            "component in arm_id. This is a missing declaration, not a "
            "fold-boundary violation."
        )
    return purpose


@dataclass(frozen=True, slots=True)
class _BatchSyncWork:
    symbol: str
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class BarCoverageObservation:
    """Timestamp-only availability metadata from a provider bar response."""

    provider: str
    symbol: str
    timeframe: str
    requested_start: str
    requested_end: str
    row_count: int
    unique_timestamp_count: int
    duplicate_timestamp_count: int
    outside_requested_window_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    timestamp_sha256: str

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-safe record that cannot expose OHLCV values."""
        return {
            "provider": self.provider,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "row_count": self.row_count,
            "unique_timestamp_count": self.unique_timestamp_count,
            "duplicate_timestamp_count": self.duplicate_timestamp_count,
            "outside_requested_window_count": self.outside_requested_window_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "timestamp_sha256": self.timestamp_sha256,
        }


@dataclass(frozen=True, slots=True)
class StoredTimestampAudit:
    """Stored timestamp counts; UTC dates do not establish exchange sessions."""

    coverage: BarCoverageObservation
    utc_date_counts: tuple[tuple[str, int], ...]

    @property
    def anomalies(self) -> tuple[str, ...]:
        findings = []
        if self.coverage.duplicate_timestamp_count:
            findings.append("duplicate_timestamps")
        if self.coverage.outside_requested_window_count:
            findings.append("out_of_scope_timestamps")
        if self.coverage.timeframe == "1d" and any(count > 1 for _, count in self.utc_date_counts):
            findings.append("multiple_daily_rows_per_utc_date")
        return tuple(findings)


def _summarize_bar_coverage(
    frame: pl.DataFrame,
    *,
    provider: str,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
) -> BarCoverageObservation:
    """Discard market values and summarize only returned UTC timestamps."""
    if frame.is_empty():
        timestamps: list[datetime] = []
    else:
        if "timestamp" not in frame.columns:
            raise SchemaValidationError("Missing required column: timestamp")
        timestamp_dtype = frame["timestamp"].dtype
        if not isinstance(timestamp_dtype, pl.Datetime) or timestamp_dtype.time_zone is None:
            raise SchemaValidationError("Timestamps must be timezone-aware")
        timestamps = cast(
            list[datetime],
            frame["timestamp"].dt.convert_time_zone("UTC").sort().to_list(),
        )

    unique_count = len(set(timestamps))
    outside_count = sum(value.date() < start or value.date() > end for value in timestamps)
    canonical_timestamps = "\n".join(value.isoformat() for value in timestamps).encode()
    return BarCoverageObservation(
        provider=provider.lower(),
        symbol=symbol.upper(),
        timeframe=timeframe,
        requested_start=start.isoformat(),
        requested_end=end.isoformat(),
        row_count=len(timestamps),
        unique_timestamp_count=unique_count,
        duplicate_timestamp_count=len(timestamps) - unique_count,
        outside_requested_window_count=outside_count,
        first_timestamp=timestamps[0].isoformat() if timestamps else None,
        last_timestamp=timestamps[-1].isoformat() if timestamps else None,
        timestamp_sha256=hashlib.sha256(canonical_timestamps).hexdigest(),
    )


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
    _CORPORATE_ACTION_FACTORIES: ClassVar[
        dict[str, Callable[[LiqDataSettings], CorporateActionsProvider]]
    ] = {
        "sec_edgar": create_sec_edgar_provider,
    }

    def __init__(
        self,
        settings: LiqDataSettings | None = None,
        data_root: Path | None = None,
        lockbox_ledger: LockboxLedger | None = None,
        research_log_root: Path | None = None,
    ) -> None:
        """Initialize DataService.

        Args:
            settings: Optional settings instance (uses get_settings() if not provided)
            data_root: Optional override for data storage directory
            lockbox_ledger: Optional research lockbox ledger for declared-purpose reads
            research_log_root: Optional audit directory for immutable source stores
        """
        self._settings = settings or get_settings()
        self._data_root = data_root or self._settings.data_root
        self._research_log_root = research_log_root or self._data_root
        self._store = ParquetStore(str(self._data_root))
        self._lockbox_ledger = lockbox_ledger
        self._lockbox_guard: LockboxGuard | None = None
        self._corporate_actions_provider_cache: dict[str, CorporateActionsProvider] = {}
        self._bar_coverage_provider_cache: dict[str, MarketDataProvider] = {}

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
        """Lockbox guard using the configured research audit directory."""
        if self._lockbox_guard is None:
            if self._lockbox_ledger is None:
                self._lockbox_guard = LockboxGuard(
                    usage_log_path=self._research_log_root / USAGE_LOG_FILENAME
                )
            else:
                self._lockbox_guard = LockboxGuard(
                    usage_log_path=self._research_log_root / USAGE_LOG_FILENAME,
                    ledger=self._lockbox_ledger,
                )
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
        asset_class: str | None = None,
        timeframe: str | None = None,
        data_kind: str = "bars",
    ) -> None:
        """Route a read through the lockbox guard; refuse an undeclared one.

        There is no path through this method that serves a read with no
        declared ``purpose``. The public entry points refuse first, so the
        refusal here is the backstop for any future internal caller.

        There is also no path that serves a read leaving no record. A read the
        active ledger governs goes to ``assert_period_allowed``; one it does not
        goes to :meth:`_guard_ungoverned_read`, which refuses an unclassified
        provider under a research purpose and logs everything it serves. The
        guarantee the declaration buys -- that the read is visible -- did not
        previously hold for a provider outside the mapping, however correct the
        declaration was.
        """
        declared_purpose = _require_declared_purpose(purpose, "this read")
        if asset_class == "short_interest_book":
            if provider.lower() != "tradestation":
                raise LockboxViolationError(
                    "short_interest_book coverage authorization admits TradeStation only"
                )
            if data_kind != "bar_coverage":
                raise LockboxViolationError(
                    "short_interest_book research access is a timestamp-only coverage probe only"
                )
            if timeframe != "1d":
                raise LockboxViolationError(
                    "short_interest_book coverage research reads admit daily bars only "
                    "(timeframe='1d')"
                )
        if asset_class == "index_recon_book" and data_kind == "bars" and timeframe != "1d":
            raise LockboxViolationError(
                "index_recon_book research reads admit daily bars only (timeframe='1d')"
            )
        if not arm_id:
            raise ValueError("arm_id is required when a research purpose is declared")
        dataset = self.lockbox_guard.classify_read(
            provider, symbol, asset_class=asset_class, data_kind=data_kind
        )
        if dataset is None:
            self._guard_ungoverned_read(
                provider,
                start,
                end,
                purpose=declared_purpose,
                arm_id=arm_id,
                final_portfolio_review=final_portfolio_review,
            )
            return
        self.lockbox_guard.assert_period_allowed(
            dataset,
            start,
            end,
            purpose=declared_purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
        )

    def _guard_ungoverned_read(
        self,
        provider: str,
        start: date | None,
        end: date | None,
        *,
        purpose: str,
        arm_id: str,
        final_portfolio_review: bool,
    ) -> None:
        """Refuse or record a read no ledger dataset governs.

        The provider is either deliberately ungoverned -- on
        :data:`~liq.data.lockbox.UNGOVERNED_PROVIDERS` with a stated reason -- or
        unclassified, meaning nobody has decided. A research read of an
        unclassified provider is refused here, before any store or provider is
        touched, because serving it is what let a declared research read escape
        every control and leave no trace.

        The refusal is scoped to research purposes deliberately. ``dev_smoke``
        is served and logged, which keeps ingestion, fixture builds and
        diagnostics working, and leaves ``_get_provider``'s more specific
        "Unknown provider" diagnostic in front of a genuinely misspelled
        provider name rather than displacing it with a governance message.
        """
        if purpose in RESEARCH_PURPOSES and provider.lower() not in UNGOVERNED_PROVIDERS:
            raise UnclassifiedProviderError(
                f"provider '{provider}' is classified by nothing: it is outside "
                "resolve_dataset's ledger mapping and is not on the ungoverned "
                f"allow-list {sorted(UNGOVERNED_PROVIDERS)}, so a '{purpose}' read of "
                "it cannot be checked against any fold window. This is neither a "
                "missing declaration nor a fold-boundary violation: nobody has "
                "classified this provider. Classifying it is a governance decision, "
                "not a code change; a read that is not research declares 'dev_smoke' "
                "and is served and logged."
            )
        self.lockbox_guard.record_ungoverned_read(
            provider,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
        )

    def _storage_key(self, provider: str, symbol: str, timeframe: str) -> str:
        """Build storage key for bars via provider/key_builder."""
        return f"{provider}/{key_builder.bars(symbol, timeframe)}"

    @staticmethod
    def _assert_storage_key_holds_one_venue(
        storage_provider: str, provider: MarketDataProvider
    ) -> None:
        """Refuse an ingest that would write a venue's bars under another's key.

        Called from every path that writes provider bars under a caller-chosen
        key: ``fetch``, ``backfill``, ``sync`` and ``sync_batch``. Enforcing it
        on some of them would leave the same hole in a different shape.

        One venue, one key. The storage key comes from the caller's argument and
        the venue from ``settings``, so the two are chosen independently and the
        store cannot tell them apart afterwards: bars carry no venue provenance,
        and the only documented way the mismatch has ever surfaced is an
        engineering note written seven months before anybody acted on it.

        The declared venue for a key defaults to the key's own name, so this is
        ordinarily provider-name equality. ``STORAGE_KEY_VENUES`` records the one
        measured exception -- the ``binance`` tree holds Binance.US bars -- which
        makes a truthful Binance.US write pass and a Binance.com write under the
        same key fail as market mixing.

        A provider that does not report a ``str`` name is a test double, not a
        venue; the check has nothing to compare and skips rather than inventing
        an answer. That tolerance and the one name it depends on are both
        pinned in ``TestOneVenueOneStorageKey``, so neither can quietly grow
        into a hole.
        """
        contacted = getattr(provider, "name", None)
        if not isinstance(contacted, str):
            return
        declared = declared_storage_venue(storage_provider)
        if contacted.lower() == declared:
            return
        raise StorageVenueMismatchError(
            f"storage key '{storage_provider.lower()}' is declared to hold "
            f"'{declared}' bars, but the constructed provider contacted "
            f"'{contacted.lower()}'. Writing them under this key would put two "
            "venues in one dataset, which nothing downstream could detect: "
            "stored bars carry no venue provenance. Fix the provider settings "
            "or the key; do not copy or alias bars between keys."
        )

    def _keep_bars_before_span(self, storage_key: str, aggregated: pl.DataFrame) -> pl.DataFrame:
        """Prepend cached bars that predate a rebuilt aggregate's span.

        A rebuild regenerates only the span its 1m base covers, so cached bars
        older than that span (e.g. daily history reaching back further than 1m
        coverage) must survive the write instead of being truncated away.
        """
        span_start = cast(datetime, aggregated["timestamp"].min())
        cached = self._store.read(storage_key, end=span_start.date())
        history = cached.filter(pl.col("timestamp") < span_start)
        if history.is_empty():
            return aggregated
        return pl.concat([history, aggregated], how="diagonal_relaxed").sort("timestamp")

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

    def _get_corporate_actions_provider(self, provider_name: str) -> CorporateActionsProvider:
        """Get a provider through its narrow corporate-action capability."""
        normalized_name = provider_name.lower()
        cached = self._corporate_actions_provider_cache.get(normalized_name)
        if cached is not None:
            return cached
        factory = self._CORPORATE_ACTION_FACTORIES.get(normalized_name)
        if factory is not None:
            provider = factory(self._settings)
            self._corporate_actions_provider_cache[normalized_name] = provider
            return provider
        provider = self._get_provider(provider_name)
        if not isinstance(provider, CorporateActionsProvider):
            raise ValueError(f"Provider {provider_name} does not support corporate actions")
        self._corporate_actions_provider_cache[normalized_name] = provider
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
        asset_class: str | None = ...,
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
        asset_class: str | None = ...,
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
        asset_class: str | None = None,
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
            purpose: Required research purpose of the read: one of
                "discovery", "validation", "characterization",
                "forward_accrual", or "dev_smoke" for reads that are not
                research. Routes the read through the lockbox guard.
            arm_id: Required research arm, or the component name for a
                ``dev_smoke`` read
            final_portfolio_review: Human-only flag permitting program-lockbox
                reads; recorded in the usage log
            asset_class: Optional asset class discriminating fold governance for
                reads that share a provider (e.g. "future" routes TradeStation
                ``@MBT``/``@MET`` to the crypto-futures ledger, not the equity
                cohort). When omitted, the provider/symbol mapping applies.

        Returns:
            DataFrame with OHLCV data

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                store is touched)
            FileNotFoundError: If data doesn't exist
            LockboxViolationError: If a declared-purpose read violates the
                lockbox ledger fold boundaries

        Example:
            >>> ds = DataService()
            >>> df = ds.load(
            ...     "oanda", "EUR_USD", "1m", purpose="dev_smoke", arm_id="demo"
            ... )
        """
        _require_declared_purpose(purpose, "DataService.load")
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
            asset_class=asset_class,
            timeframe=timeframe,
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
                            rebuilt = self._keep_bars_before_span(storage_key, aggregated)
                            self._store.write(storage_key, rebuilt, mode="overwrite")
        if not self._store.exists(storage_key):
            if timeframe != "1m":
                base_key = self._storage_key(provider, symbol, "1m")
                if self._store.exists(base_key):
                    if batch_size is not None:
                        raise StoredBarsRequiredError(
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
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
    ) -> Iterator[pl.DataFrame]:
        """Iterate over data in batches for memory-efficient processing.

        Refuses an undeclared read on first iteration -- which is the moment
        this generator would otherwise touch the store -- so no batch is ever
        served without a declaration.
        """
        _require_declared_purpose(purpose, "DataService.iter_batches")
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

    def _validate_timestamp_inventory(self, key: str, files: tuple[PinnedParquetFile, ...]) -> None:
        base = self._data_root / key
        ancestry = (self._data_root, *base.parents)
        if base.is_symlink() or any(path.is_symlink() for path in ancestry):
            raise SchemaValidationError("timestamp metadata key is redirected")
        expected_paths: set[str] = set()
        for item in files:
            parts = item.path[len(key) + 1 :].split("/")
            expected_paths.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
        for path in base.rglob("*"):
            relative = path.relative_to(base).as_posix()
            if path.is_symlink() or relative not in expected_paths:
                raise SchemaValidationError(
                    "timestamp metadata inventory contains an unexpected path"
                )
            expected_paths.remove(relative)
        if expected_paths:
            raise SchemaValidationError("timestamp metadata inventory contains missing paths")

    def audit_pinned_timestamps(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        *,
        expected: tuple[TimestampFileMetadata, ...],
        purpose: str | None,
        arm_id: str,
        max_files: int,
        max_rows: int,
        limits: TimestampProjectionLimits,
    ) -> TimestampDomainAudit:
        """Guard and exhaust pinned timestamp projections before publishing a domain audit."""
        _require_declared_purpose(purpose, "DataService.audit_pinned_timestamps")
        if start > end:
            raise ValueError("timestamp domain start must not follow end")
        key = self._storage_key(provider, symbol, timeframe)
        files = tuple(item.identity for item in expected)
        if any(not item.path.startswith(key + "/") for item in files):
            raise ValueError("timestamp domain files must match declared key")
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=False,
            timeframe=timeframe,
            data_kind="timestamp_domain",
        )
        inventory_hash = hashlib.sha256(
            json.dumps([(item.path, item.sha256) for item in files], separators=(",", ":")).encode()
        ).hexdigest()
        attempt_id = uuid.uuid4().hex
        accumulator: TimestampDomainAccumulator | None = None

        def record(event: str, **details: str | int) -> None:
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "attempt_id": attempt_id,
                "access_kind": "timestamp_domain",
                "event": event,
                "provider": provider,
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "purpose": purpose,
                "arm_id": arm_id,
                "inventory_sha256": inventory_hash,
                **details,
            }
            with (self._research_log_root / "timestamp_domain_usage.jsonl").open("a") as output:
                output.write(json.dumps(entry, sort_keys=True) + "\n")

        record("attempt", files=len(files))
        try:
            if any(type(value) is not int or value <= 0 for value in (max_files, max_rows)):
                raise TimestampDomainRefused(None, None, 0, None, 0, "invalid domain budgets")
            if not 0 < len(files) <= max_files or len({item.path for item in files}) != len(files):
                raise TimestampDomainRefused(
                    None, None, 0, None, 0, "file budget or duplicate path"
                )
            if sum(item.row_count for item in expected) > max_rows:
                raise TimestampDomainRefused(
                    None, None, 0, None, 0, "declared key row budget exceeded"
                )
            lower = (start - date(1970, 1, 1)).days * 86400 * 10**9
            upper = ((end - date(1970, 1, 1)).days + 1) * 86400 * 10**9
            for item in expected:
                for group in item.row_groups:
                    if group.minimum_ns < lower or group.maximum_ns >= upper:
                        raise TimestampDomainRefused(
                            item.identity,
                            group.index,
                            0,
                            None,
                            0,
                            "declared footer outside requested dates",
                        )
            self._validate_timestamp_inventory(key, files)
            accumulator = TimestampDomainAccumulator(
                start, end, daily=timeframe == "1d", max_rows=max_rows
            )
            for item in expected:
                projection = iter_pinned_timestamps(self._data_root, item, limits)
                try:
                    for batch in projection:
                        accumulator.observe(batch)
                finally:
                    projection.close()
            result = accumulator.finish()
            if result.rows != sum(item.row_count for item in expected) or result.files != len(
                files
            ):
                raise TimestampDomainRefused(
                    None, None, 0, None, result.rows, "physical domain declaration mismatch"
                )
            record(
                "completed",
                files=result.files,
                rows=result.rows,
                timestamp_sha256=result.timestamp_sha256,
            )
            return result
        except (TimestampMetadataRefused, SchemaValidationError, OSError) as exc:
            record(
                "refused",
                error_type=type(exc).__name__,
                reason=str(exc),
                partial_rows=accumulator.rows if accumulator else 0,
            )
            raise

    def preflight_stored_timestamps(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        *,
        files: tuple[PinnedParquetFile, ...],
        purpose: str | None,
        arm_id: str,
        max_files: int,
        max_file_bytes: int,
        max_row_groups: int,
    ) -> tuple[TimestampFileMetadata, ...]:
        """Guard pinned footer declarations; success does not authorize page decoding."""
        _require_declared_purpose(purpose, "DataService.preflight_stored_timestamps")
        if start > end:
            raise ValueError("timestamp metadata start must not follow end")
        key = self._storage_key(provider, symbol, timeframe)
        if any(not item.path.startswith(key + "/") for item in files):
            raise ValueError("timestamp metadata files must match the declared key")
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=False,
            timeframe=timeframe,
            data_kind="timestamp_metadata",
        )
        inventory_hash = hashlib.sha256(
            json.dumps([(item.path, item.sha256) for item in files], separators=(",", ":")).encode()
        ).hexdigest()
        attempt_id = uuid.uuid4().hex

        def record(event: str, **details: str | int) -> None:
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "attempt_id": attempt_id,
                "access_kind": "timestamp_footer",
                "event": event,
                "provider": provider,
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "purpose": purpose,
                "arm_id": arm_id,
                "inventory_sha256": inventory_hash,
                **details,
            }
            with (self._research_log_root / "timestamp_metadata_usage.jsonl").open("a") as output:
                output.write(json.dumps(entry, sort_keys=True) + "\n")

        record("attempt", files=len(files))
        try:
            budgets = (max_files, max_file_bytes, max_row_groups)
            if any(type(value) is not int or value <= 0 for value in budgets):
                raise TimestampMetadataRefused(
                    "timestamp metadata budgets must be positive integers"
                )
            if len(files) > max_files:
                raise TimestampMetadataRefused("timestamp metadata file budget exceeded")
            self._validate_timestamp_inventory(key, files)
            metadata = inspect_timestamp_metadata(
                self._data_root,
                files,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_row_groups=max_row_groups,
            )
            epoch = date(1970, 1, 1)
            lower = (start - epoch).days * 86_400 * 10**9
            upper = ((end - epoch).days + 1) * 86_400 * 10**9
            for item in metadata:
                for group in item.row_groups:
                    if group.minimum_ns < lower or group.maximum_ns >= upper:
                        raise SchemaValidationError(
                            f"timestamp metadata outside requested dates: {item.identity.path} "
                            f"group={group.index} min_ns={group.minimum_ns} max_ns={group.maximum_ns}"
                        )
            record("completed", files=len(metadata), rows=sum(item.row_count for item in metadata))
            return metadata
        except (TimestampMetadataRefused, SchemaValidationError, OSError) as exc:
            record("refused", error_type=type(exc).__name__, reason=str(exc))
            raise

    def audit_stored_timestamps(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
        *,
        purpose: str | None,
        arm_id: str,
        batch_size: int,
        max_rows: int,
    ) -> StoredTimestampAudit:
        """Audit timestamp-only stored batches, without provider calls or rollups.

        Inputs must already be an admitted, physically bounded store snapshot.
        Date filters do not restrict which partition files the store enumerates.
        The loader sorts batches, so this audit makes no original-order claim.
        A row-budget refusal is incomplete evidence, never a truncated summary.
        """
        if start > end:
            raise ValueError("timestamp audit start must not follow end")
        if any(type(value) is not int or value <= 0 for value in (batch_size, max_rows)):
            raise ValueError("timestamp audit budgets must be positive integers")
        frames: list[pl.DataFrame] = []
        rows = 0
        for frame in self.iter_batches(
            provider,
            symbol,
            timeframe,
            start,
            end,
            columns=["timestamp"],
            purpose=purpose,
            arm_id=arm_id,
            batch_size=min(batch_size, max_rows),
            final_portfolio_review=False,
        ):
            if frame.columns != ["timestamp"]:
                raise SchemaValidationError("Timestamp audit requires only the timestamp column")
            dtype = frame.schema["timestamp"]
            if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
                raise SchemaValidationError("Timestamp audit requires UTC datetime schema")
            if frame["timestamp"].null_count():
                raise SchemaValidationError("Timestamp audit contains null timestamps")
            rows += frame.height
            if rows > max_rows:
                raise TimestampAuditLimitError("Timestamp audit row budget exceeded")
            frames.append(frame)
        if not frames:
            raise SchemaValidationError("No timestamp schema observed; empty read is unverified")
        timestamps = pl.concat(frames)
        coverage = _summarize_bar_coverage(
            timestamps,
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        counts = Counter(value.isoformat() for value in timestamps["timestamp"].dt.date())
        return StoredTimestampAudit(
            coverage=coverage, utc_date_counts=tuple(sorted(counts.items()))
        )

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
        dataset: str | None = None,
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
        asset_class: str | None = None,
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
            dataset: Optional provider dataset override (Databento only), for
                securities reachable only on a specific venue dataset (e.g. a
                delisted name on ``XNAS.ITCH``). Ignored by providers that do
                not accept it.
            purpose: Required purpose of the provider read; fetches route
                through the same lockbox guard as stored reads. Ingestion
                declares ``dev_smoke``.
            arm_id: Required research arm, or the component name for a
                ``dev_smoke`` fetch.
            final_portfolio_review: Human-only program-lockbox override.
            asset_class: Optional fold discriminator, such as ``future`` for
                TradeStation crypto futures.

        Returns:
            DataFrame with fetched OHLCV data

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                provider is contacted)

        Example:
            >>> from datetime import date
            >>> ds = DataService()
            >>> df = ds.fetch(
            ...     "oanda", "EUR_USD", date(2024, 1, 1), date(2024, 1, 31),
            ...     purpose="dev_smoke", arm_id="demo",
            ... )
        """
        _require_declared_purpose(purpose, "DataService.fetch")
        if end is None:
            end = date.today()

        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
            asset_class=asset_class,
            timeframe=timeframe,
        )

        prov = self._get_provider(provider)
        self._assert_storage_key_holds_one_venue(provider, prov)
        extra = {"dataset": dataset} if dataset is not None else {}
        df = prov.fetch_bars(symbol, start, end, timeframe=timeframe, **extra)
        validate_ohlc(df)

        if save:
            storage_key = self._storage_key(provider, symbol, timeframe)
            self._store.write(storage_key, df, mode=mode)

        return df

    def probe_bar_coverage(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date,
        *,
        timeframe: str,
        purpose: str,
        arm_id: str,
        asset_class: str,
    ) -> BarCoverageObservation:
        """Fetch bars but expose only timestamp-level availability metadata.

        This API intentionally has no save mode and never returns its provider
        frame. It is suitable for governed coverage characterization where
        market values must not reach the calling research harness.
        """
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=False,
            asset_class=asset_class,
            timeframe=timeframe,
            data_kind="bar_coverage",
        )
        normalized_provider = provider.lower()
        coverage_provider = self._bar_coverage_provider_cache.get(normalized_provider)
        if coverage_provider is None:
            coverage_provider = self._get_provider(provider)
            self._bar_coverage_provider_cache[normalized_provider] = coverage_provider
        frame = coverage_provider.fetch_bars(symbol, start, end, timeframe=timeframe)
        return _summarize_bar_coverage(
            frame,
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

    def _prepare_oanda_capture(self, window: SourcePriceWindow) -> None:
        """Capture-local owners may bind ingress before the ordinary guard."""

    # The default retains the keyword contract without interpreting capture-local scope.
    def _get_oanda_capture_provider(
        self,
        window: SourcePriceWindow,  # noqa: ARG002
    ) -> MarketDataProvider:
        """Resolve the capture route without changing ordinary provider selection."""
        return self._get_provider("oanda")

    def capture_oanda_source_prices(
        self, scope: SourcePriceWindow, *, purpose: str | None, arm_id: str
    ) -> SourceBody | PriceCaptureFailure:
        """Guard an exact discovery case before structurally quarantined BA transport."""
        _require_declared_purpose(purpose, "DataService.capture_oanda_source_prices")
        if purpose != "discovery" or arm_id != PROBE_ARM:
            raise PriceCapturePolicyRefused("DECLARATION_REFUSED")
        window = SourcePriceWindow.model_validate(scope.model_dump())
        self._prepare_oanda_capture(window)
        if (
            self.lockbox_guard.classify_read(
                "oanda", window.symbol, asset_class="forex", data_kind="bars"
            )
            != "oanda_fx"
        ):
            raise PriceCapturePolicyRefused("CLASSIFICATION_REFUSED")
        self._guard_research_read(
            "oanda",
            window.symbol,
            window.day,
            window.day,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=False,
            asset_class="forex",
            timeframe="1m",
            data_kind="bars",
        )
        source = self._get_oanda_capture_provider(window)
        if not isinstance(source, SourcePriceCaptureProvider):
            raise PriceCapturePolicyRefused("PROVIDER_UNSUPPORTED")
        return source.capture_source_prices(window)

    def _prepare_oanda_cohort_capture(self, window: CohortSourceWindow) -> None:
        """Cohort owners may bind original one-use ingress before the actual guard."""

    def _get_oanda_cohort_capture_provider(
        self,
        window: CohortSourceWindow,  # noqa: ARG002
    ) -> MarketDataProvider:
        """Select the retained cohort capability independently of legacy capture."""
        return self._get_provider("oanda")

    def capture_oanda_cohort_prices(
        self, scope: CohortSourceWindow, *, purpose: str | None, arm_id: str
    ) -> SourceBody | PriceCaptureFailure:
        """Guard a closed development case before resolving its cohort provider."""
        from liq.data.oanda_cohort_capture import capture_guarded_cohort_prices

        declaration = _require_declared_purpose(purpose, "DataService.capture_oanda_cohort_prices")
        return capture_guarded_cohort_prices(self, scope, purpose=declaration, arm_id=arm_id)

    def probe_oanda_source_day(
        self, scope: CandleDay, *, purpose: str | None, arm_id: str
    ) -> CandleDayObservation:
        """Guard the actual day before a fixed account-scoped metadata request."""
        _require_declared_purpose(purpose, "DataService.probe_oanda_source_day")
        if purpose != "discovery" or arm_id != PROBE_ARM:
            raise ValidationError("Unapproved OANDA source-verification declaration")
        validate_source_day(scope)
        self._guard_research_read(
            "oanda",
            scope.symbol,
            scope.day,
            scope.day,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=False,
            asset_class="forex",
            timeframe="1m",
            data_kind="bar_coverage",
        )
        source = self._get_provider("oanda")
        if not isinstance(source, AccountSourceDayProvider):
            raise ProviderError("Provider does not support account day probes")
        return source.probe_source_day(scope)

    def probe_oanda_instrument_metadata(
        self, request: InstrumentMetadataRequest
    ) -> InstrumentMetadataObservation:
        """Audit a declared current-instrument query without fabricating a historical read."""
        from liq.data.oanda_instrument_audit import collect_instrument_metadata

        _require_declared_purpose(request.purpose, "DataService.probe_oanda_instrument_metadata")
        return collect_instrument_metadata(self, request)

    def probe_oanda_account_candle(
        self,
        *,
        purpose: str | None,
        arm_id: str,
    ) -> AccountCandleObservation:
        """Guard the fixed historical compatibility minute and expose metadata only.

        This is a source capability, not execution permission. The caller must
        provide the separately reviewed one-use claim and end-to-end deadline.
        """
        day = PROBE_START.date()
        self._guard_research_read(
            "oanda",
            "EUR_USD",
            day,
            day,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=False,
            asset_class="forex",
            timeframe="1m",
            data_kind="bar_coverage",
        )
        source = self._get_provider("oanda")
        if not isinstance(source, AccountCandleProbeProvider):
            raise ProviderError("Provider does not support account candle probes")
        return source.probe_account_candle()

    def fetch_bid_ask_bars(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date,
        *,
        timeframe: str = "1m",
        purpose: str | None = None,
        arm_id: str | None = None,
        asset_class: str | None = None,
    ) -> pl.DataFrame:
        """Fetch separate bid/ask OHLC through the research guard.

        The result is never written to the midpoint bar store. Historical quoted
        sides include their quoted spread; they do not establish execution fills.
        Both date bounds are inclusive. Source admission remains the caller's
        research responsibility, in addition to the enforced period declaration.
        """
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=False,
            asset_class=asset_class,
            timeframe=timeframe,
        )
        source = self._get_provider(provider)
        if not isinstance(source, BidAskBarProvider):
            raise ProviderError(f"Provider {provider} does not support bid/ask candles")
        return source.fetch_bid_ask_bars(symbol, start, end, timeframe=timeframe)

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
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
        asset_class: str | None = None,
    ) -> list[Any]:
        """Fetch corporate actions through the research guard.

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                provider is contacted).
        """
        _require_declared_purpose(purpose, "DataService.fetch_corporate_actions")
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
            asset_class=asset_class,
            data_kind="corporate_actions",
        )
        prov = self._get_corporate_actions_provider(provider)
        actions = prov.get_corporate_actions(symbol, start, end)
        if save and actions:
            key = f"{provider}/{key_builder.corp_actions(symbol)}"
            self._store.write(key, pl.DataFrame(actions), mode="overwrite")
        return actions

    def fetch_security_master(
        self,
        provider: str,
        symbols: tuple[str, ...],
        start: date,
        end: date,
        *,
        max_records: int,
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
    ) -> list[SecurityMasterRecord]:
        _require_declared_purpose(purpose, "DataService.fetch_security_master")
        validate_security_master_request(symbols, start, end, max_records)
        if (
            provider.lower() != "databento"
            or purpose != "characterization"
            or arm_id != "classical_cross_sectional_ranker"
            or final_portfolio_review
        ):
            raise LockboxViolationError("security-master source scope is not authorized")
        for symbol in symbols:
            self._guard_research_read(
                provider,
                symbol,
                start,
                end,
                purpose=purpose,
                arm_id=arm_id,
                final_portfolio_review=False,
                data_kind="security_master",
            )
        source = self._get_provider(provider)
        if not isinstance(source, SecurityMasterProvider):
            raise ValueError(f"Provider {provider} does not support security master")
        return source.get_security_master(symbols, start, end, max_records=max_records)

    def fetch_alpaca_source_page(
        self,
        request: AlpacaBarsRequest
        | AlpacaCorporateActionsRequest
        | AlpacaDailyFixtureRequest
        | AlpacaCalendarRequest
        | AlpacaMomentumCalendarRequest
        | AlpacaMomentumDailyRequest
        | AlpacaAnnouncementCalendarRequest
        | AlpacaAnnouncementDailyRequest
        | AlpacaEventBasisDailyRequest,
        *,
        page_token: str | None,
        purpose: str | None,
        arm_id: str | None,
        final_portfolio_review: bool = False,
        on_http_attempt: Callable[[], None] | None = None,
    ) -> bytes:
        _require_declared_purpose(purpose, "DataService.fetch_alpaca_source_page")
        event_basis = isinstance(request, AlpacaEventBasisDailyRequest)
        announcement = isinstance(request, (AlpacaAnnouncementCalendarRequest, AlpacaAnnouncementDailyRequest))
        if isinstance(request, AlpacaEventBasisDailyRequest):
            request = validate_event_basis_request(request)
        if isinstance(request, (AlpacaAnnouncementCalendarRequest, AlpacaAnnouncementDailyRequest)):
            request = validate_announcement_request(request)
        momentum = isinstance(request, (AlpacaMomentumCalendarRequest, AlpacaMomentumDailyRequest))
        if isinstance(request, AlpacaMomentumCalendarRequest):
            request = AlpacaMomentumCalendarRequest.model_validate(request)
        elif isinstance(request, AlpacaMomentumDailyRequest):
            request = AlpacaMomentumDailyRequest.model_validate(request)
        daily = isinstance(request, (AlpacaDailyFixtureRequest, AlpacaMomentumDailyRequest, AlpacaAnnouncementDailyRequest, AlpacaEventBasisDailyRequest))
        calendar = isinstance(request, (AlpacaCalendarRequest, AlpacaMomentumCalendarRequest, AlpacaAnnouncementCalendarRequest))
        expected_ledger = (
            CLASSICAL_RANKER_EVENT_BASIS_FIXTURE_LEDGER_V1
            if event_basis
            else CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
            if announcement
            else CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1
            if momentum
            else CLASSICAL_RANKER_CALENDAR_REFERENCE_LEDGER_V1
            if calendar
            else CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1
            if daily
            else CLASSICAL_RANKER_ALPACA_LEDGER_V1
        )
        if (
            self._lockbox_ledger != expected_ledger
            or purpose != "characterization"
            or arm_id != "classical_cross_sectional_ranker"
            or final_portfolio_review
            or not isinstance(
                request,
                (
                    AlpacaBarsRequest,
                    AlpacaCorporateActionsRequest,
                    AlpacaDailyFixtureRequest,
                    AlpacaCalendarRequest,
                    AlpacaMomentumCalendarRequest,
                    AlpacaMomentumDailyRequest,
                    AlpacaAnnouncementCalendarRequest,
                    AlpacaAnnouncementDailyRequest,
                    AlpacaEventBasisDailyRequest,
                ),
            )
            or ((daily or calendar) and page_token is not None)
        ):
            raise LockboxViolationError("Alpaca source request is outside its dedicated ledger")
        if isinstance(request, (AlpacaCalendarRequest, AlpacaMomentumCalendarRequest, AlpacaAnnouncementCalendarRequest)):
            symbols = ("US_MARKET",)
            start = request.start
            end = request.end
            data_kind = (
                "ranker_announcement_calendar_reference"
                if announcement
                else "ranker_momentum_calendar_reference"
                if momentum
                else "ranker_calendar_reference"
            )
        elif isinstance(
            request,
            (AlpacaBarsRequest, AlpacaDailyFixtureRequest, AlpacaMomentumDailyRequest, AlpacaAnnouncementDailyRequest, AlpacaEventBasisDailyRequest),
        ):
            symbols = (request.symbol,)
            start = request.start.date()
            end = request.end.date()
            data_kind = (
                f"ranker_event_basis_{request.adjustment}_fixture"
                if event_basis
                else "ranker_announcement_daily_fixture"
                if announcement
                else "ranker_momentum_daily_fixture"
                if momentum
                else "ranker_daily_fixture"
                if daily
                else "ranker_bars"
            )
        else:
            symbols = request.symbols
            start = request.start
            end = request.end
            data_kind = "ranker_corporate_actions"
        for symbol in symbols:
            self._guard_research_read(
                "alpaca",
                symbol,
                start,
                end,
                purpose=purpose,
                arm_id=arm_id,
                final_portfolio_review=False,
                data_kind=data_kind,
            )
        source = self._get_provider("alpaca")
        if not isinstance(source, AlpacaProvider):
            raise ValueError("Alpaca provider source capability is not concrete")
        if on_http_attempt is not None:
            return source.fetch_source_page(
                request, page_token=page_token, on_http_attempt=on_http_attempt
            )
        return source.fetch_source_page(request, page_token=page_token)

    def fetch_adjustment_factors(
        self,
        provider: str,
        symbol: str,
        start: date,
        end: date,
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
        asset_class: str | None = None,
    ) -> list[Any]:
        """Fetch adjustment factors through the research guard.

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                provider is contacted).
        """
        _require_declared_purpose(purpose, "DataService.fetch_adjustment_factors")
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
            asset_class=asset_class,
            data_kind="adjustment_factors",
        )
        prov = self._get_provider(provider)
        if not hasattr(prov, "get_adjustment_factors"):
            raise ValueError(f"Provider {provider} does not support adjustment factors")
        return prov.get_adjustment_factors(symbol, start, end)

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
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
        final_portfolio_review: bool = False,
        asset_class: str | None = None,
    ) -> pl.DataFrame:
        """Fetch only missing ranges based on gaps in stored data.

        Returns market values, so it carries the same declaration as any other
        read. The guard runs before the stored series is read and before any
        gap-filling branch reaches the provider directly rather than through
        :meth:`fetch`; ingestion declares ``dev_smoke``. The empty-store branch
        delegates to :meth:`fetch`, which guards again, so that branch leaves
        two ledger entries for the same window -- over-reporting a read is the
        safe direction for an append-only usage log.

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                store or provider is touched).
        """
        _require_declared_purpose(purpose, "DataService.backfill")
        self._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
            asset_class=asset_class,
            timeframe=timeframe,
        )
        storage_key = self._storage_key(provider, symbol, timeframe)
        existing = (
            self._store.read(storage_key) if self._store.exists(storage_key) else pl.DataFrame()
        )

        if existing.is_empty():
            return self.fetch(
                provider,
                symbol,
                start,
                end,
                timeframe=timeframe,
                save=True,
                mode="append",
                purpose=purpose,
                arm_id=arm_id,
                final_portfolio_review=final_portfolio_review,
                asset_class=asset_class,
            )

        prov = self._get_provider(provider)
        self._assert_storage_key_holds_one_venue(provider, prov)

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

    def validate(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate data integrity.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe
            purpose: Required purpose of the underlying read. This reads the
                whole stored series, so it is unbounded and admits
                ``dev_smoke`` only; a research purpose is refused for want of
                bounds rather than given invented ones.
            arm_id: Required component or arm name, recorded in the usage log

        Returns:
            Dict with validation results:
                - valid: Whether data passes validation
                - row_count: Number of rows
                - null_count: Number of null values
                - errors: List of error messages

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                store is touched)

        Example:
            >>> ds = DataService()
            >>> result = ds.validate(
            ...     "oanda", "EUR_USD", "1m", purpose="dev_smoke", arm_id="demo"
            ... )
            >>> result["valid"]
            True
        """
        _require_declared_purpose(purpose, "DataService.validate")
        df = self.load(provider, symbol, timeframe, purpose=purpose, arm_id=arm_id)

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

    def info(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
    ) -> dict[str, Any]:
        """Get data metadata.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe
            purpose: Required purpose of the underlying read. Unbounded, so it
                admits ``dev_smoke`` only.
            arm_id: Required component or arm name, recorded in the usage log

        Returns:
            Dict with metadata:
                - row_count: Number of rows
                - columns: List of column names
                - start: First timestamp (date)
                - end: Last timestamp (date)

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                store is touched)

        Example:
            >>> ds = DataService()
            >>> info = ds.info(
            ...     "oanda", "EUR_USD", "1m", purpose="dev_smoke", arm_id="demo"
            ... )
            >>> info["row_count"]
            1000000
        """
        _require_declared_purpose(purpose, "DataService.info")
        storage_key = self._storage_key(provider, symbol, timeframe)
        df = self.load(provider, symbol, timeframe, purpose=purpose, arm_id=arm_id)

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

    def stats(
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
    ) -> dict[str, Any]:
        """Get statistical summary of data.

        Args:
            provider: Provider name
            symbol: Symbol name
            timeframe: Timeframe
            purpose: Required purpose of the underlying read. Unbounded, so it
                admits ``dev_smoke`` only.
            arm_id: Required component or arm name, recorded in the usage log

        Returns:
            Dict with statistics for each numeric column:
                - mean, min, max, std, count

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                store is touched)

        Example:
            >>> ds = DataService()
            >>> stats = ds.stats(
            ...     "oanda", "EUR_USD", "1m", purpose="dev_smoke", arm_id="demo"
            ... )
            >>> stats["close"]["mean"]
            1.0875
        """
        _require_declared_purpose(purpose, "DataService.stats")
        df = self.load(provider, symbol, timeframe, purpose=purpose, arm_id=arm_id)

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
        self,
        provider: str,
        symbol: str,
        timeframe: str,
        expected_minutes: int,
        *,
        purpose: str | None = None,
        arm_id: str | None = None,
    ) -> list[tuple[Any, Any]]:
        """Detect gaps for a symbol/timeframe using expected minutes.

        Reads the whole stored series, so it is unbounded and admits
        ``dev_smoke`` only.

        Raises:
            UndeclaredReadError: If no purpose is declared (raised before the
                store is touched).
        """
        _require_declared_purpose(purpose, "DataService.gaps")
        df = self.load(provider, symbol, timeframe, purpose=purpose, arm_id=arm_id)
        if df.is_empty():
            return []
        return detect_gaps(df, timedelta(minutes=expected_minutes))

    def validate_credentials(self, provider: str) -> bool:
        """Validate provider credentials via provider adapter."""
        prov = self._get_provider(provider)
        return prov.validate_credentials()

    # ----- universe resolution / sync ----------------------------------

    def _universe_resolver(self, named_universes: Any) -> Any:
        """Resolver whose constituent source is point-in-time when snapshots exist.

        ``source="snapshot"`` uses :class:`DirectorySnapshotSource`
        (``pit=True``) rooted at
        ``{data_root}/reference/universes/snapshots``. ``source="stub"`` is the
        only route to :class:`InMemoryStubSource` (``pit=False``), which
        downstream sweeps reject to avoid survivorship bias. A snapshot source
        missing its directory or requested id file is a fail-closed error,
        never a silent current-membership fallback.
        """
        from liq.data.universes import (
            DirectorySnapshotSource,
            InMemoryStubSource,
            UniverseResolver,
        )

        snapshot_dir = self._data_root / "reference" / "universes" / "snapshots"
        return UniverseResolver(
            constituent_sources={
                "snapshot": DirectorySnapshotSource(snapshot_dir),
                "stub": InMemoryStubSource(),
            },
            named_universes=named_universes,
        )

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
            UniverseDefinition,
            UniverseKind,
            UniverseRegistry,
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

        return self._universe_resolver(named_universes).resolve(definition, as_of=as_of)

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
        invoking the billable :meth:`sync`. Supported timeframes are derived
        from the Databento provider's schema registry.

        Returns a dict with keys:
            billable_bytes, estimated_cost_usd, dataset, schema,
            symbols, start, end, provider_request_id

        Raises ``ValueError`` on bad timeframe, empty universe, inverted
        date range, or missing ``DATABENTO_API_KEY``.
        """
        schema = SCHEMA_BY_TIMEFRAME.get(timeframe)
        if schema is None:
            raise ValueError(
                "estimate_databento_cost supports timeframes "
                f"{sorted(SCHEMA_BY_TIMEFRAME)} (got {timeframe!r})"
            )
        if end < start:
            raise ValueError(f"end ({end}) must be on or after start ({start})")
        resolved = self.resolve_universe(universe, as_of=end, registry=registry)
        symbols = list(resolved.symbols)
        if not symbols:
            raise ValueError(
                "estimate_databento_cost requires a non-empty symbols list "
                "after universe resolution"
            )

        client = self._databento_historical_client()
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

    def get_databento_dataset_range(self, *, dataset: str) -> dict[str, Any]:
        """Return Databento's non-billable dataset-range metadata payload."""
        if not dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        payload = self._databento_historical_client().metadata.get_dataset_range(dataset=dataset)
        return dict(payload)

    def _databento_historical_client(self) -> Any:
        """Build a metadata-capable client without touching billable endpoints."""
        if not self._settings.databento_api_key:
            raise ValueError(
                "DATABENTO_API_KEY not configured. Set it in .env or pass "
                "settings with `databento_api_key=...`."
            )
        # Lazy import keeps the databento dependency off the hot path.
        import databento  # noqa: PLC0415

        return databento.Historical(key=self._settings.databento_api_key)

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
            UniverseDefinition,
            UniverseRegistry,
        )

        if max_workers <= 0:
            raise ValueError(f"max_workers must be >= 1; got {max_workers}")

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

        resolved = self._universe_resolver(named_universes).resolve(definition, as_of=as_of or end)

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
            self._assert_storage_key_holds_one_venue(provider, prov)
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
            UniverseDefinition,
            UniverseRegistry,
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

        resolved = self._universe_resolver(named_universes).resolve(definition, as_of=as_of or end)

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
            self._assert_storage_key_holds_one_venue(provider, prov)
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

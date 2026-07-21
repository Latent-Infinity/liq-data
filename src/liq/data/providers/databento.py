"""Databento provider for US equities 1m / 1d historical bars.

Implements the ``BaseProvider`` contract for the Databento venue with
deterministic integer-to-Decimal normalization, batch-vs-sync routing
by date span, and symbology persistence. Universes, the coverage
manifest, and ``sync(universe)`` are out of scope for this module;
they live alongside ``UniverseDefinition`` and the manifest in
``liq.data.universes`` / ``liq.data.manifest``.

Design notes
------------

* **Dataset routing.** The provider maps ``(asset_class, timeframe)`` to
  a Databento dataset code via :data:`DATASET_ROUTING`. The current map
  covers equities; other asset classes raise. The mapping is data, not
  control flow, so futures and crypto slot in later without protocol
  changes.
* **Batch vs. sync.** Date ranges shorter than
  :data:`DATABENTO_BATCH_THRESHOLD_DAYS` use ``timeseries.get_range``
  (synchronous streaming); longer ranges go through the batch download
  API. The threshold is a constructor argument so tests can flip it.
* **Decimal precision.** Databento ships OHLCV prices as int64 scaled
  by 1e9 (DBN-internal "q9" convention). The requirements call for
  no float round-trip; the provider converts via
  ``Decimal(int) / DATABENTO_PRICE_SCALE`` (an exact rational divide)
  and stores at the standard liq-store schema scale of 8 decimal places.
* **Symbology persistence.** Each fetch appends one row per
  ``raw_symbol → instrument_id`` mapping the response carries, under
  the ``reference/databento/symbology`` key. Downstream queries through
  raw symbol can then look up the right ``instrument_id`` for an as-of
  window even if a symbol was renamed mid-history.
* **Logging.** Every fetch emits one INFO record with
  ``event=databento_fetch`` and queryable fields including ``provider,
  dataset, symbols_count, start, end, request_kind, bytes_in,
  duration_ms, sync_run_id``. The record uses ``extra=`` so the fields
  are queryable attributes on the ``LogRecord`` rather than baked into
  the message.
* **Network policy.** No real Databento call ever happens unless the
  caller supplies a real ``databento.Historical`` instance. Tests
  inject a fake client and never hit the network; the real-API smoke
  test is opt-in via ``RUN_DATABENTO=1``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import polars as pl
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

from liq.data.exceptions import DataQualityError, ProviderError, ProviderNoDataError
from liq.data.protocols import BatchJob
from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE, BaseProvider
from liq.data.sync_events import (
    EVENT_BATCH_DOWNLOAD_STARTED,
    EVENT_BATCH_POLLING,
    EVENT_BATCH_RESUMED,
    EVENT_BATCH_SUBMITTED,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from liq.store.protocols import TimeSeriesStore

_logger = logging.getLogger(__name__)

# ----- module-level constants -----------------------------------------------

DATASET_ROUTING: dict[tuple[str, str], str] = {
    ("equity", "1m"): "EQUS.MINI",
    ("equity", "1d"): "EQUS.SUMMARY",
}
"""``(asset_class, timeframe) → Databento dataset code``.

The mapping is data, not code structure: adding futures /
``GLBX.MDP3`` later is a one-line entry, not a protocol change.
"""

DATABENTO_BATCH_THRESHOLD_DAYS: int = 14
"""Date spans at or above this many calendar days route to the batch API.

Below threshold, ``timeseries.get_range`` (sync streaming) is cheaper +
faster — and bounded enough to keep the call synchronous.
"""

DATABENTO_PRICE_SCALE: Decimal = Decimal(10**9)
"""Databento DBN-internal price scaling (1 unit = $1e-9).

Prices arrive as int64; we divide by this exact rational and store as
``Decimal`` with the schema scale of 8 places. **No float intermediate.**
"""

DATABENTO_UNDEF_PRICE: int = 9223372036854775807
"""Databento's documented sentinel for an undefined / no-trade price
(``INT64_MAX``).

The SDK ships it as ``databento_dbn.UNDEF_PRICE`` (re-exported as
``databento.UNDEF_PRICE``); the equality is asserted by a regression test
so this literal keeps the module importable without paying the SDK's
import cost on the hot path. A bar carrying this value in any OHLC field
has no valid price: blindly scaling it via
``Decimal(int) / DATABENTO_PRICE_SCALE`` would emit a ~$9.2e9 phantom
price, so such bars are **excluded** (never imputed) at conversion time.
Real equity prices are orders of magnitude below the sentinel's q9 value
(~$9.2e9), so the guard cannot drop a genuine price.
"""

SYMBOLOGY_KEY = "reference/databento/symbology"
"""``liq-store`` key prefix for the symbology mapping table."""

SCHEMA_BY_TIMEFRAME: dict[str, str] = {
    "1m": "ohlcv-1m",
    "1d": "ohlcv-1d",
}
"""``timeframe → Databento schema name``."""


DATABENTO_DEFAULT_MAX_RETRY_ATTEMPTS: int = 3
"""Number of attempts (initial + retries) for transient-error retries."""

DATABENTO_DEFAULT_BACKOFF_BASE_SECONDS: float = 1.0
"""Base for exponential backoff between retries."""


DATABENTO_DEFAULT_BATCH_JOBS_DIR: str = os.path.join(
    tempfile.gettempdir(), "liq-data", "databento-batch-jobs"
)
"""Default cache directory for batch-job resume markers.

Markers are tiny JSON sidecars that survive a process restart so an
interrupted batch download can resume on next invocation without
re-submitting (re-billing) the job.
"""


DATABENTO_DEFAULT_BATCH_POLL_SECONDS: float = 5.0
"""Default delay between ``get_job_details`` polls while a batch job is
running. Overridable for tests + impatient callers."""


# ----- exceptions ------------------------------------------------------------


class DatabentoError(ProviderError):
    """Base class for Databento-specific errors."""


class DatabentoTransientError(DatabentoError):
    """Retry-eligible failure (5xx, connection blip, transient network).

    ``retry_after`` is honored over the provider's exponential backoff
    when present; venue-supplied hints (e.g. the ``Retry-After`` header
    on a 429) take precedence over local policy.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after: float | None = retry_after


class DatabentoRateLimitError(DatabentoTransientError):
    """429 from Databento. Carries the ``Retry-After`` hint when supplied."""


class DatabentoSchemaError(DatabentoError):
    """The response schema did not match the requested schema.

    Not retry-eligible — retrying will return the same mismatched store.
    """


class AggregateCrossCheckError(DataQualityError):
    """``validate_aggregate`` found a 1m-aggregate vs. 1d mismatch larger
    than the configured tolerance.

    The instance carries a per-field diff so consumers can surface which
    OHLCV component drifted and by how much.
    """

    def __init__(
        self,
        message: str,
        *,
        symbol: str,
        date_iso: str,
        diffs: dict[str, dict[str, Decimal]],
    ) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.date_iso = date_iso
        self.diffs = diffs


AGGREGATE_TOLERANCE: Decimal = Decimal("0.00001")  # 0.001 %
"""Price-field tolerance as a fraction of midrange close."""


AGGREGATION_WINDOWS: dict[str, str] = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
}
"""Supported timeframes for local 1m → nm aggregation, mapped to Polars
``group_by_dynamic`` ``every`` strings. The keys are the public API; the
values are the implementation detail."""


# ----- records ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BatchJobMarker:
    """Sidecar state for a batch job in flight.

    The marker is written under :data:`DATABENTO_DEFAULT_BATCH_JOBS_DIR`
    (or a caller-supplied override) keyed by a SHA-256 hash of the
    canonical request fields. If a download is interrupted, the marker
    survives; the next invocation with the same request signature
    resumes the existing job instead of re-submitting.

    Markers are deleted on successful completion.
    """

    signature: str
    job_id: str
    submitted_at: float
    dataset: str
    symbol: str
    schema: str
    start_iso: str
    end_iso: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "job_id": self.job_id,
            "submitted_at": self.submitted_at,
            "dataset": self.dataset,
            "symbol": self.symbol,
            "schema": self.schema,
            "start_iso": self.start_iso,
            "end_iso": self.end_iso,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _BatchJobMarker:
        return cls(
            signature=str(data["signature"]),
            job_id=str(data["job_id"]),
            submitted_at=float(data["submitted_at"]),
            dataset=str(data["dataset"]),
            symbol=str(data["symbol"]),
            schema=str(data["schema"]),
            start_iso=str(data["start_iso"]),
            end_iso=str(data["end_iso"]),
        )


@dataclass(frozen=True, slots=True)
class DatabentoBarRecord:
    """One Databento OHLCV msg, normalized to the fields we actually use.

    Prices are kept as the raw int64 ``q9`` representation; the
    conversion to ``Decimal`` happens once, at the polars-DataFrame
    materialization step, so we never round-trip through float.
    """

    ts_event_ns: int
    instrument_id: int
    open_q9: int
    high_q9: int
    low_q9: int
    close_q9: int
    volume: int
    symbol: str


# ----- normalization helpers -------------------------------------------------


_PRICE_QUANT = Decimal("1e-8")


def _q9_to_decimal(value_q9: int) -> Decimal:
    """Convert a q9 int to a ``Decimal`` quantized to 8 places of scale.

    The conversion is exact at every step:

    1. ``Decimal(int)`` is exact.
    2. ``/ DATABENTO_PRICE_SCALE`` is exact (rational divide).
    3. ``.quantize(_PRICE_QUANT)`` truncates the (essentially unused)
       9th place to match ``PRICE_DTYPE``'s declared scale of 8 places.

    No float arithmetic, no string parsing — Polars's ``Decimal`` column
    accepts ``Decimal`` instances directly with bit-exact preservation.
    """
    return (Decimal(value_q9) / DATABENTO_PRICE_SCALE).quantize(_PRICE_QUANT)


def _has_undef_price(record: DatabentoBarRecord) -> bool:
    """True when any OHLC field carries Databento's UNDEF_PRICE sentinel.

    The bar is a no-trade artifact and must be excluded rather than
    scaled (scaling ``INT64_MAX`` q9 would emit a ~$9.2e9 phantom price).
    Volume is deliberately not checked — only prices carry the sentinel.
    """
    return DATABENTO_UNDEF_PRICE in (
        record.open_q9,
        record.high_q9,
        record.low_q9,
        record.close_q9,
    )


_DBN_FILE_SUFFIXES = (".dbn", ".dbn.zst", ".dbn.gz")


class _MergedDBNStore:
    """Normalize downloaded DBN path returns.

    ``Databento.HistoricalBatch.download(...)`` can return a single
    path, a directory path, or a sequence of paths depending on SDK
    version and extraction shape. Older fakes returned a single
    store-like object instead.

    Non-DBN files (manifest.json, symbology.json) in the download list
    are skipped; ``DBNStore.from_file`` would otherwise raise on them.
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = [p for p in paths if str(p).endswith(_DBN_FILE_SUFFIXES)]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._paths)

    @classmethod
    def from_download_result(cls, raw: Any) -> Any:
        """Wrap path-shaped SDK returns, or leave store-shaped fakes alone."""
        paths = cls._coerce_paths(raw)
        if paths is not None:
            return cls(paths)
        return raw

    @staticmethod
    def _coerce_paths(raw: Any) -> list[Path] | None:
        if isinstance(raw, str | os.PathLike):
            path = Path(raw)
            return sorted(path.rglob("*")) if path.is_dir() else [path]
        if isinstance(raw, list | tuple | set) and all(
            isinstance(p, str | os.PathLike) for p in raw
        ):
            out: list[Path] = []
            for item in raw:
                path = Path(item)
                if path.is_dir():
                    out.extend(sorted(path.rglob("*")))
                else:
                    out.append(path)
            return out
        return None


@dataclass(frozen=True)
class _MaterializedDBNStore:
    """In-memory snapshot of a batch download's records + symbology.

    Built inside :meth:`DatabentoProvider._invoke_batch` while the
    staging directory is still alive; once the staging
    ``TemporaryDirectory`` exits, the source ``.dbn.zst`` files are
    gone. The snapshot satisfies the downstream duck-typed contract
    (``__iter__``, ``.symbology``, ``.metadata``) so
    ``_extract_records`` / ``_persist_symbology`` /
    ``_verify_response_schema`` work without further changes.
    """

    records: tuple[DatabentoBarRecord, ...]
    symbology: dict[str, Any]
    schema_name: str | None

    def __iter__(self) -> Any:
        return iter(self.records)

    @property
    def metadata(self) -> Any:
        class _Meta:
            pass

        m = _Meta()
        m.schema = self.schema_name  # type: ignore[attr-defined]
        return m

    @property
    def schema(self) -> str | None:
        return self.schema_name


def _records_to_dataframe(records: list[DatabentoBarRecord]) -> pl.DataFrame:
    """Build a typed polars DataFrame from a list of Databento bar records.

    Always returns the canonical liq-store bar schema:

    * ``timestamp``: ``Datetime("us", "UTC")``
    * ``open``/``high``/``low``/``close``: ``PRICE_DTYPE`` (Decimal 38, 8)
    * ``volume``: ``VOLUME_DTYPE`` (Decimal 38, 2)
    * ``symbol``: ``Utf8``
    * ``instrument_id``: ``UInt32``

    Empty input still returns a frame with the right schema (so callers
    can ``.is_empty()`` without column-existence checks).
    """
    schema: dict[str, Any] = {
        "timestamp": pl.Datetime("us", "UTC"),
        "symbol": pl.Utf8,
        "instrument_id": pl.UInt32,
        "open": PRICE_DTYPE,
        "high": PRICE_DTYPE,
        "low": PRICE_DTYPE,
        "close": PRICE_DTYPE,
        "volume": VOLUME_DTYPE,
    }

    if not records:
        return pl.DataFrame(schema=schema)

    # Exclude no-trade bars whose price fields carry Databento's
    # UNDEF_PRICE sentinel (INT64_MAX). Scaling the sentinel would leak a
    # ~$9.2e9 phantom price into stored OHLCV; missing ⇒ exclusion, not
    # imputation. Excluded counts are logged per symbol so the drop is
    # observable in the run's telemetry.
    valid_records: list[DatabentoBarRecord] = []
    excluded_by_symbol: dict[str, int] = {}
    for r in records:
        if _has_undef_price(r):
            excluded_by_symbol[r.symbol] = excluded_by_symbol.get(r.symbol, 0) + 1
        else:
            valid_records.append(r)

    for symbol, count in excluded_by_symbol.items():
        _logger.info(
            "databento undef-price bars excluded %s %d",
            symbol,
            count,
            extra={
                "event": "databento_undef_price_excluded",
                "provider": "databento",
                "symbol": symbol,
                "count": count,
            },
        )

    if not valid_records:
        return pl.DataFrame(schema=schema)

    rows = [
        {
            "timestamp": datetime.fromtimestamp(r.ts_event_ns / 1e9, tz=UTC),
            "symbol": r.symbol,
            "instrument_id": int(r.instrument_id),
            "open": _q9_to_decimal(r.open_q9),
            "high": _q9_to_decimal(r.high_q9),
            "low": _q9_to_decimal(r.low_q9),
            "close": _q9_to_decimal(r.close_q9),
            "volume": Decimal(int(r.volume)),
        }
        for r in valid_records
    ]
    return pl.DataFrame(rows, schema=schema)


def aggregate_bars(df_1m: pl.DataFrame, *, timeframe: str) -> pl.DataFrame:
    """Roll 1-minute bars up to a coarser timeframe locally.

    Coarser bars are computed from 1m locally rather than re-purchased.
    Aggregation rules per bucket:

    * ``open``  = first row's ``open``
    * ``high``  = max of all ``high`` in the bucket
    * ``low``   = min of all ``low`` in the bucket
    * ``close`` = last row's ``close``
    * ``volume`` = sum of ``volume``
    * ``symbol`` = pass-through (group preserves it)
    * ``instrument_id`` = first row's id (assumed stable per symbol/day)

    Bucket alignment uses Polars ``group_by_dynamic`` with ``every``
    matching ``timeframe``; daily buckets respect calendar-day
    boundaries in UTC. Empty input returns an empty frame with the
    same schema as the input.
    """
    if timeframe not in AGGREGATION_WINDOWS:
        raise ValueError(
            f"unsupported timeframe for aggregation: {timeframe!r}; "
            f"supported: {sorted(AGGREGATION_WINDOWS)}"
        )
    if df_1m.is_empty():
        return df_1m

    every = AGGREGATION_WINDOWS[timeframe]
    group_keys = ["symbol"] if "symbol" in df_1m.columns else []

    grouped = (
        df_1m.sort("timestamp")
        .group_by_dynamic(
            "timestamp",
            every=every,
            closed="left",
            group_by=group_keys or None,
            label="left",
        )
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            (
                pl.col("instrument_id").first().alias("instrument_id")
                if "instrument_id" in df_1m.columns
                else pl.lit(None).alias("instrument_id")
            ),
        )
    )

    # Reorder columns so callers see the canonical layout.
    canonical = ["timestamp", "symbol", "instrument_id", "open", "high", "low", "close", "volume"]
    present = [c for c in canonical if c in grouped.columns]
    return grouped.select(present)


def _symbology_to_dataframe(symbology: dict[str, Any]) -> pl.DataFrame:
    """Flatten a Databento symbology dict to a long frame.

    Accepts two shapes:

    * **Real ``DBNStore.symbology``** — a metadata-rich dict whose
      instrument-id mappings live under ``["mappings"]``::

          {"symbols": [...], "stype_in": ..., ...,
           "mappings": {"SPY": [{"start_date": date(...), "end_date": date(...),
                                  "symbol": "15144"}]}}

      Here ``symbol`` is the instrument-id rendered as a string and the
      dict-key is the raw venue symbol.

    * **Test/simple shape** — a flat ``{raw_symbol: [{"instrument_id":
      int, "valid_from": str, "valid_to": str}, ...]}`` dict. Easier to
      write fixtures against and round-trips through the same output
      schema.

    Either way we emit a long Polars frame with columns
    ``raw_symbol | instrument_id | valid_from | valid_to``. Empty input
    or unrecognized shapes return an empty frame with the right schema.
    """
    schema: dict[str, Any] = {
        "raw_symbol": pl.Utf8,
        "instrument_id": pl.UInt32,
        "valid_from": pl.Utf8,
        "valid_to": pl.Utf8,
    }

    # The real DBNStore.symbology nests the mapping table under
    # "mappings"; the test/simple shape is already the mapping table
    # itself. Detect by membership.
    if "mappings" in symbology and isinstance(symbology["mappings"], dict):
        mapping_table: dict[str, Any] = symbology["mappings"]
        instrument_key = "symbol"
        valid_from_key = "start_date"
        valid_to_key = "end_date"
    else:
        mapping_table = symbology
        instrument_key = "instrument_id"
        valid_from_key = "valid_from"
        valid_to_key = "valid_to"

    rows = []
    for raw_symbol, mappings in mapping_table.items():
        if not isinstance(mappings, list):
            continue
        for m in mappings:
            if not isinstance(m, dict):
                continue
            iid_raw = m.get(instrument_key)
            if iid_raw is None:
                continue
            try:
                iid = int(iid_raw)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "raw_symbol": str(raw_symbol),
                    "instrument_id": iid,
                    "valid_from": str(m.get(valid_from_key, "")),
                    "valid_to": str(m.get(valid_to_key, "")),
                }
            )
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _estimate_bytes_in(*, store: Any, rows: int) -> int:
    """Best-effort inbound byte count for structured fetch logs."""
    for attr in ("nbytes", "num_bytes", "bytes_in"):
        raw = getattr(store, attr, None)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return max(rows, 0) * 64


def _read_status_code(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from SDK exceptions."""
    candidates = [exc, getattr(exc, "response", None)]
    for obj in candidates:
        if obj is None:
            continue
        for attr in ("status_code", "http_status", "status"):
            raw = getattr(obj, attr, None)
            if raw is None:
                continue
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
    return None


def _is_no_data_error(exc: Exception) -> bool:
    """Return true for Databento terminal 422 no-data/symbology misses."""
    if _read_status_code(exc) != 422:
        return False
    text_parts = [str(exc)]
    for attr in ("json_body", "http_body", "body", "content"):
        raw = getattr(exc, attr, None)
        if raw is None:
            continue
        if isinstance(raw, bytes):
            text_parts.append(raw.decode("utf-8", errors="ignore"))
        else:
            text_parts.append(str(raw))
    text = " ".join(text_parts).lower()
    return any(
        needle in text
        for needle in (
            "data_no_data_found_for_request",
            "symbology_invalid_request",
            "no data was found",
            "none of the symbols could be resolved",
        )
    )


def _read_retry_after_seconds(exc: Exception) -> float | None:
    """Read ``Retry-After`` as seconds from an SDK or HTTP exception."""
    candidates = [exc, getattr(exc, "response", None)]
    for obj in candidates:
        if obj is None:
            continue
        headers = getattr(obj, "headers", None)
        if headers is None:
            continue
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after is None:
            continue
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            try:
                retry_dt = parsedate_to_datetime(str(retry_after))
            except (TypeError, ValueError):
                continue
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=UTC)
            return max((retry_dt.astimezone(UTC) - datetime.now(UTC)).total_seconds(), 0.0)
    return None


def _looks_like_network_error(exc: Exception) -> bool:
    """Identify common SDK transport exceptions without binding to one SDK."""
    cls = type(exc)
    marker = f"{cls.__module__}.{cls.__name__}".lower()
    network_tokens = (
        "connectionerror",
        "connecterror",
        "connecttimeout",
        "readtimeout",
        "timeout",
        "networkerror",
        "transporterror",
    )
    return any(token in marker for token in network_tokens)


def _date_range(start: date, end: date) -> list[date]:
    """Inclusive date list for per-session validation."""
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def _filter_day(df: pl.DataFrame, when: date) -> pl.DataFrame:
    """Return bars whose UTC timestamp falls on ``when``."""
    if df.is_empty():
        return df
    start_dt = datetime(when.year, when.month, when.day, tzinfo=UTC)
    end_dt = start_dt + timedelta(days=1)
    return df.filter((pl.col("timestamp") >= start_dt) & (pl.col("timestamp") < end_dt))


def _databento_raw_symbol(symbol: str) -> str:
    """Convert common class-share notation to Databento raw_symbol form."""
    return symbol.replace("-", ".")


def _same_databento_symbol(left: str, right: str) -> bool:
    return _databento_raw_symbol(left).upper() == _databento_raw_symbol(right).upper()


# ----- provider --------------------------------------------------------------


class DatabentoProvider(BaseProvider):
    """Databento ``DataProvider`` implementation (US equities, 1m / 1d).

    Args:
        api_key: Databento API key. Required even when ``client`` is
            injected — it's the source of truth for which credential
            future code paths use (e.g. the batch ``download`` step).
        asset_class: Routing key into :data:`DATASET_ROUTING`. Defaults
            to ``"equity"``; expansion is one mapping entry away.
        batch_threshold_days: Spans at or above this many days route to
            the batch API. Defaults to
            :data:`DATABENTO_BATCH_THRESHOLD_DAYS`.
        client: Optional pre-built ``databento.Historical`` (or duck-
            typed fake for tests). Production code passes ``None`` and
            the provider builds a real client lazily.
        store: Optional ``TimeSeriesStore`` for symbology persistence.
            When ``None``, symbology rows are computed and logged but
            **not** persisted (useful for short-lived ad-hoc fetches).

    Raises:
        ValueError: if ``api_key`` is empty.
    """

    DATA_BASE_URL = "https://hist.databento.com"  # documentation only

    def __init__(
        self,
        api_key: str,
        *,
        asset_class: str = "equity",
        batch_threshold_days: int = DATABENTO_BATCH_THRESHOLD_DAYS,
        client: Any | None = None,
        store: TimeSeriesStore | None = None,
        max_retry_attempts: int = DATABENTO_DEFAULT_MAX_RETRY_ATTEMPTS,
        backoff_base_seconds: float = DATABENTO_DEFAULT_BACKOFF_BASE_SECONDS,
        batch_jobs_dir: str | None = None,
        batch_poll_seconds: float = DATABENTO_DEFAULT_BATCH_POLL_SECONDS,
        sleep_fn: Any | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for Databento provider")
        if max_retry_attempts < 1:
            raise ValueError(f"max_retry_attempts must be >= 1, got {max_retry_attempts}")
        if backoff_base_seconds < 0:
            raise ValueError(f"backoff_base_seconds must be >= 0, got {backoff_base_seconds}")
        self._api_key = api_key
        self.asset_class = asset_class
        self.batch_threshold_days = batch_threshold_days
        self._client = client
        self._store = store
        self.max_retry_attempts = max_retry_attempts
        self.backoff_base_seconds = backoff_base_seconds
        self.batch_jobs_dir = Path(
            batch_jobs_dir if batch_jobs_dir is not None else DATABENTO_DEFAULT_BATCH_JOBS_DIR
        )
        self.batch_poll_seconds = batch_poll_seconds
        self._sleep_fn = sleep_fn if sleep_fn is not None else time.sleep

    # ----- BaseProvider properties --------------------------------------

    @property
    def name(self) -> str:
        return "databento"

    @property
    def supported_asset_classes(self) -> list[str]:
        return sorted({a for (a, _t) in DATASET_ROUTING})

    @property
    def supported_timeframes(self) -> list[str]:
        return sorted({t for (a, t) in DATASET_ROUTING if a == self.asset_class})

    # ----- fetch_bars ---------------------------------------------------

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
        *,
        validate_aggregate: bool = False,
        dataset: str | None = None,
    ) -> pl.DataFrame:
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        # ``dataset`` overrides ``DATASET_ROUTING`` for securities that only
        # exist on a venue dataset (e.g. delisted names on XNAS.ITCH).
        dataset = dataset if dataset is not None else self._resolve_dataset(timeframe)
        schema = self._resolve_schema(timeframe)

        span_days = (end - start).days + 1
        request_kind = "batch" if span_days >= self.batch_threshold_days else "get_range"

        start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC)

        # ``sync_run_id`` is the correlation key threaded through every
        # ``databento_retry`` and the final ``databento_fetch`` event so
        # log readers can stitch the attempt sequence together.
        sync_run_id = str(uuid4())

        t0 = time.monotonic()
        store = self._call_with_retry(
            lambda: self._invoke_remote(
                request_kind=request_kind,
                dataset=dataset,
                symbol=symbol,
                schema=schema,
                start=start_dt,
                end=end_dt,
            ),
            sync_run_id=sync_run_id,
            dataset=dataset,
            symbol=symbol,
            request_kind=request_kind,
        )
        self._verify_response_schema(store, expected=schema)
        records = self._extract_records(store, fallback_symbol=symbol)
        df = _records_to_dataframe(records)
        self._persist_symbology(store)
        if validate_aggregate and timeframe == "1m":
            for day in _date_range(start, end):
                self.validate_aggregate(
                    symbol,
                    day,
                    df_1m=_filter_day(df, day),
                )
        duration_ms = int((time.monotonic() - t0) * 1000)
        bytes_in = _estimate_bytes_in(store=store, rows=df.height)

        _logger.info(
            "databento fetch %s/%s %d records",
            dataset,
            symbol,
            df.height,
            extra={
                "event": "databento_fetch",
                "provider": self.name,
                "dataset": dataset,
                "symbols_count": 1,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "request_kind": request_kind,
                "bytes_in": bytes_in,
                "duration_ms": duration_ms,
                "sync_run_id": sync_run_id,
            },
        )
        return df

    # ----- external batch orchestration ---------------------------------

    def submit_batch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
        *,
        dataset: str,
        sync_run_id: str | None = None,
    ) -> BatchJob:
        """Submit or resume a Databento batch bars job without waiting."""
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        schema = self._resolve_schema(timeframe)
        start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC)
        return self._call_with_retry(
            lambda: self._submit_or_resume_batch_job(
                client=self._get_client(),
                dataset=dataset,
                symbol=symbol,
                schema=schema,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
            ),
            sync_run_id=sync_run_id or str(uuid4()),
            dataset=dataset,
            symbol=symbol,
            request_kind="batch_submit",
        )

    def poll_batch_bars(
        self,
        job: BatchJob,
        *,
        sync_run_id: str | None = None,
    ) -> bool:
        """Return True when a submitted Databento batch job is ready."""
        return bool(
            self._call_with_retry(
                lambda: self._poll_batch_job(self._get_client(), job),
                sync_run_id=sync_run_id or str(uuid4()),
                dataset=job.dataset,
                symbol=job.symbol,
                request_kind="batch_poll",
            )
        )

    def fetch_completed_batch_bars(
        self,
        job: BatchJob,
        *,
        sync_run_id: str | None = None,
    ) -> pl.DataFrame:
        """Download and materialize a completed Databento batch job."""
        t0 = time.monotonic()
        store = self._call_with_retry(
            lambda: self._download_batch_job(self._get_client(), job),
            sync_run_id=sync_run_id or str(uuid4()),
            dataset=job.dataset,
            symbol=job.symbol,
            request_kind="batch_download",
        )
        expected_schema = str(job.metadata.get("schema") or self._resolve_schema(job.timeframe))
        self._verify_response_schema(store, expected=expected_schema)
        records = self._extract_records(store, fallback_symbol=job.symbol)
        df = _records_to_dataframe(records)
        self._persist_symbology(store)
        duration_ms = int((time.monotonic() - t0) * 1000)
        bytes_in = _estimate_bytes_in(store=store, rows=df.height)
        _logger.info(
            "databento fetch %s/%s %d records",
            job.dataset,
            job.symbol,
            df.height,
            extra={
                "event": "databento_fetch",
                "provider": self.name,
                "dataset": job.dataset,
                "symbols_count": 1,
                "start": job.start.date().isoformat(),
                "end": job.end.date().isoformat(),
                "request_kind": "batch",
                "bytes_in": bytes_in,
                "duration_ms": duration_ms,
                "sync_run_id": sync_run_id or "",
            },
        )
        return df

    # ----- retry + schema verification ----------------------------------

    def _call_with_retry(
        self,
        op: Any,
        *,
        sync_run_id: str,
        dataset: str,
        symbol: str,
        request_kind: str,
    ) -> Any:
        """Invoke ``op`` with bounded retry on ``DatabentoTransientError``.

        Backoff strategy:

        * If the exception carries ``retry_after``, honor it verbatim —
          venue hints (e.g. 429 ``Retry-After``) take precedence.
        * Otherwise use exponential backoff: ``base * 2^(attempt-1)``.

        Non-transient exceptions propagate immediately without retry.
        Each retry emits a ``databento_retry`` log event sharing the
        caller's ``sync_run_id`` for correlation.
        """
        retrying = Retrying(
            stop=stop_after_attempt(self.max_retry_attempts),
            retry=retry_if_exception_type(DatabentoTransientError),
            wait=self._retry_wait_seconds,
            sleep=self._sleep_fn,
            before_sleep=lambda state: self._log_retry(
                state,
                sync_run_id=sync_run_id,
                dataset=dataset,
                symbol=symbol,
                request_kind=request_kind,
            ),
            reraise=True,
        )
        return retrying(lambda: self._invoke_with_transient_translation(op))

    def _invoke_with_transient_translation(self, op: Any) -> Any:
        try:
            return op()
        except DatabentoTransientError:
            raise
        except Exception as exc:
            if _is_no_data_error(exc):
                raise ProviderNoDataError(str(exc)) from exc
            transient = self._coerce_transient_error(exc)
            if transient is None:
                raise
            raise transient from exc

    def _coerce_transient_error(self, exc: Exception) -> DatabentoTransientError | None:
        if isinstance(exc, DatabentoError):
            return None
        status_code = _read_status_code(exc)
        retry_after = _read_retry_after_seconds(exc)
        if status_code == 429:
            return DatabentoRateLimitError(str(exc), retry_after=retry_after)
        if status_code is not None and 500 <= status_code <= 599:
            return DatabentoTransientError(str(exc), retry_after=retry_after)
        if _looks_like_network_error(exc):
            return DatabentoTransientError(str(exc), retry_after=retry_after)
        return None

    def _retry_wait_seconds(self, retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
        if isinstance(exc, DatabentoTransientError) and exc.retry_after is not None:
            return exc.retry_after
        return self.backoff_base_seconds * (2 ** (retry_state.attempt_number - 1))

    def _log_retry(
        self,
        retry_state: RetryCallState,
        *,
        sync_run_id: str,
        dataset: str,
        symbol: str,
        request_kind: str,
    ) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome is not None else None
        _logger.info(
            "databento retry %s attempt=%d",
            dataset,
            retry_state.attempt_number,
            extra={
                "event": "databento_retry",
                "provider": self.name,
                "sync_run_id": sync_run_id,
                "dataset": dataset,
                "symbol": symbol,
                "request_kind": request_kind,
                "attempt": retry_state.attempt_number,
                "max_attempts": self.max_retry_attempts,
                "backoff_s": retry_state.next_action.sleep if retry_state.next_action else 0.0,
                "error_type": type(exc).__name__ if exc is not None else "",
                "error_message": str(exc) if exc is not None else "",
            },
        )

    def _verify_response_schema(self, store: Any, *, expected: str) -> None:
        """Raise ``DatabentoSchemaError`` when the response schema doesn't
        match the request.

        Real ``databento.DBNStore`` exposes ``schema`` as a string (or
        ``Schema`` enum coerced to one). Stores that don't expose the
        attribute are tolerated — older fakes pass through unchecked.
        """
        actual = getattr(store, "schema", None)
        if actual is None:
            return
        actual_str = str(actual)
        if actual_str != expected:
            raise DatabentoSchemaError(
                f"Databento response schema {actual_str!r} does not match "
                f"requested schema {expected!r}"
            )

    # ----- list_instruments (protocol stub) -----------------------------

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:  # noqa: ARG002
        # This provider does not ship instrument discovery; universe
        # resolution + EQUS.SUMMARY reference data live in the universe
        # machinery (separate module). Return an empty schema rather
        # than raising NotImplementedError so the protocol stays
        # consumable by callers that don't need discovery.
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "asset_class": pl.Utf8,
                "exchange": pl.Utf8,
            }
        )

    def set_store(self, store: TimeSeriesStore) -> None:
        """Attach a store for sidecar symbology persistence."""
        self._store = store

    # ----- aggregate cross-check ---------------------------------------

    def validate_aggregate(
        self,
        symbol: str,
        when: date,
        *,
        tolerance: Decimal = AGGREGATE_TOLERANCE,
        df_1m: pl.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Cross-check the local 1m-aggregated daily bar against the
        venue's ``EQUS.SUMMARY`` 1d bar for the same symbol + date.

        Raises :class:`AggregateCrossCheckError` if any OHLCV component
        drifts too far from the daily reference. Price fields use an
        absolute tolerance of ``tolerance * midrange_close`` (default
        0.001 % of the two closes' midpoint); volume must match exactly.
        Returns a small report dict on success so callers can log the
        comparison without re-fetching.

        This is the "trust but verify" workhorse for the ingestion
        pipeline — runs after a 1m fetch lands to make sure our locally
        aggregated daily matches the venue's official OHLCV.
        """
        if df_1m is None:
            df_1m = self.fetch_bars(symbol, when, when, timeframe="1m")
        df_1d_local = aggregate_bars(df_1m, timeframe="1d")
        df_1d_venue = self.fetch_bars(symbol, when, when, timeframe="1d")

        if df_1d_local.is_empty() or df_1d_venue.is_empty():
            raise AggregateCrossCheckError(
                f"validate_aggregate: empty bar(s) for {symbol} on {when.isoformat()}",
                symbol=symbol,
                date_iso=when.isoformat(),
                diffs={},
            )

        local = df_1d_local.row(0, named=True)
        venue = df_1d_venue.row(0, named=True)

        diffs: dict[str, dict[str, Decimal]] = {}
        local_close = Decimal(local["close"])
        venue_close = Decimal(venue["close"])
        midrange_close = (abs(local_close) + abs(venue_close)) / Decimal(2)
        price_tolerance = (midrange_close or Decimal(1)) * tolerance

        for field in ("open", "high", "low", "close"):
            local_val = Decimal(local[field])
            venue_val = Decimal(venue[field])
            absolute = abs(local_val - venue_val)
            if absolute > price_tolerance:
                diffs[field] = {
                    "local": local_val,
                    "venue": venue_val,
                    "absolute": absolute,
                    "allowed": price_tolerance,
                }
        local_volume = Decimal(local["volume"])
        venue_volume = Decimal(venue["volume"])
        if local_volume != venue_volume:
            diffs["volume"] = {
                "local": local_volume,
                "venue": venue_volume,
                "absolute": abs(local_volume - venue_volume),
                "allowed": Decimal(0),
            }
        if diffs:
            offenders = ", ".join(sorted(diffs))
            raise AggregateCrossCheckError(
                f"validate_aggregate: {symbol} on {when.isoformat()} drift exceeds "
                f"{tolerance} of midrange close in [{offenders}]",
                symbol=symbol,
                date_iso=when.isoformat(),
                diffs=diffs,
            )

        return {
            "matched": True,
            "symbol": symbol,
            "date": when.isoformat(),
            "tolerance": str(tolerance),
            "price_tolerance": str(price_tolerance),
        }

    # ----- internal -----------------------------------------------------

    def _resolve_dataset(self, timeframe: str) -> str:
        key = (self.asset_class, timeframe)
        if key not in DATASET_ROUTING:
            raise ProviderError(
                f"Databento has no dataset mapping for "
                f"asset_class={self.asset_class!r} timeframe={timeframe!r}"
            )
        return DATASET_ROUTING[key]

    def _resolve_schema(self, timeframe: str) -> str:
        if timeframe not in SCHEMA_BY_TIMEFRAME:
            raise ProviderError(f"Unsupported timeframe for Databento: {timeframe!r}")
        return SCHEMA_BY_TIMEFRAME[timeframe]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        # Lazy import keeps the dep cost off the import-graph hot path.
        import databento  # noqa: PLC0415

        self._client = databento.Historical(key=self._api_key)
        return self._client

    def _invoke_remote(
        self,
        *,
        request_kind: str,
        dataset: str,
        symbol: str,
        schema: str,
        start: datetime,
        end: datetime,
    ) -> Any:
        client = self._get_client()
        if request_kind == "get_range":
            return client.timeseries.get_range(
                dataset=dataset,
                symbols=[_databento_raw_symbol(symbol)],
                schema=schema,
                start=start,
                end=end,
            )
        return self._invoke_batch(
            client=client,
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            start=start,
            end=end,
        )

    # ----- batch path (with partial-recovery) ---------------------------

    @staticmethod
    def _batch_signature(
        *, dataset: str, symbol: str, schema: str, start: datetime, end: datetime
    ) -> str:
        """Hash of the canonical request fields.

        Two invocations with the same ``(dataset, symbol, schema, start, end)``
        produce the same signature → second invocation resumes the
        first's job rather than re-submitting (which would re-bill).
        """
        canonical = "|".join(
            [
                dataset,
                symbol,
                schema,
                start.isoformat(),
                end.isoformat(),
            ]
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    def _marker_path(self, signature: str) -> Path:
        return self.batch_jobs_dir / f"{signature}.json"

    def _load_marker(self, signature: str) -> _BatchJobMarker | None:
        path = self._marker_path(signature)
        if not path.exists():
            return None
        try:
            return _BatchJobMarker.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            # Corrupt marker — treat as missing so we don't get stuck.
            return None

    def _save_marker(self, marker: _BatchJobMarker) -> None:
        self.batch_jobs_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._marker_path(marker.signature).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
        tmp.replace(self._marker_path(marker.signature))

    def _delete_marker(self, signature: str) -> None:
        path = self._marker_path(signature)
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def _invoke_batch(
        self,
        *,
        client: Any,
        dataset: str,
        symbol: str,
        schema: str,
        start: datetime,
        end: datetime,
    ) -> Any:
        """Submit (or resume) a batch job, poll until done, download.

        Resume semantics:

        * If a marker exists for this request signature, skip
          ``submit_job`` and poll the existing ``job_id`` straight to
          completion. This is the load-bearing guarantee — re-submitting
          would re-bill the venue.
        * If the download itself fails (exception during ``download``),
          the marker stays on disk so the *next* invocation resumes.
        * On successful download, the marker is deleted.

        The poll loop emits no extra log events for each tick (would be
        noisy); the wrapping ``databento_fetch`` event still captures
        the duration_ms of the whole call.
        """
        job = self._submit_or_resume_batch_job(
            client=client,
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            timeframe=self._timeframe_for_schema(schema),
            start=start,
            end=end,
        )
        while not self._poll_batch_job(client, job):
            self._sleep_fn(self.batch_poll_seconds)
        return self._download_batch_job(client, job)

    def _submit_or_resume_batch_job(
        self,
        *,
        client: Any,
        dataset: str,
        symbol: str,
        schema: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> BatchJob:
        signature = self._batch_signature(
            dataset=dataset, symbol=symbol, schema=schema, start=start, end=end
        )
        marker = self._load_marker(signature)
        if marker is None:
            job = client.batch.submit_job(
                dataset=dataset,
                symbols=[_databento_raw_symbol(symbol)],
                schema=schema,
                start=start,
                end=end,
                encoding="dbn",
                stype_in="raw_symbol",
                stype_out="instrument_id",
            )
            marker = _BatchJobMarker(
                signature=signature,
                job_id=str(job.get("id") or job.get("job_id") or ""),
                submitted_at=time.time(),
                dataset=dataset,
                symbol=symbol,
                schema=schema,
                start_iso=start.isoformat(),
                end_iso=end.isoformat(),
            )
            if not marker.job_id:
                raise DatabentoError("Databento batch submission did not return a job id")
            self._save_marker(marker)
            _logger.info(
                "databento batch submitted",
                extra={
                    "event": EVENT_BATCH_SUBMITTED,
                    "provider": self.name,
                    "dataset": dataset,
                    "symbol": symbol,
                    "job_id": marker.job_id,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
        else:
            _logger.info(
                "databento batch resumed",
                extra={
                    "event": EVENT_BATCH_RESUMED,
                    "provider": self.name,
                    "dataset": dataset,
                    "symbol": symbol,
                    "job_id": marker.job_id,
                },
            )
        return BatchJob(
            provider=self.name,
            job_id=marker.job_id,
            signature=signature,
            dataset=dataset,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            metadata={"schema": schema},
        )

    def _poll_batch_job(self, client: Any, job: BatchJob) -> bool:
        terminal_states = {"done", "completed"}
        failed_states = {"canceled", "cancelled", "expired", "failed"}
        details = client.batch.get_job_details(job.job_id)
        state = str(details.get("state") or "")
        if state.lower() in terminal_states:
            return True
        if state.lower() in failed_states:
            self._delete_marker(job.signature)
            raise DatabentoError(f"Databento batch job {job.job_id!r} ended in state {state!r}")
        _logger.info(
            "databento batch polling",
            extra={
                "event": EVENT_BATCH_POLLING,
                "provider": self.name,
                "job_id": job.job_id,
                "state": state,
            },
        )
        return False

    def _download_batch_job(self, client: Any, job: BatchJob) -> Any:
        # Stage the download into a per-call ``TemporaryDirectory`` so
        # the raw ``.dbn.zst`` files are auto-deleted on exit (success
        # OR exception). The marker stays on the persistent path so a
        # mid-flight failure can still resume without re-billing.
        with tempfile.TemporaryDirectory(prefix="liq-data-databento-") as staging:
            _logger.info(
                "databento batch download started",
                extra={
                    "event": EVENT_BATCH_DOWNLOAD_STARTED,
                    "provider": self.name,
                    "job_id": job.job_id,
                    "output_dir": staging,
                },
            )
            raw = client.batch.download(job_id=job.job_id, output_dir=staging)
            # Real SDK returns ``list[Path]`` of extracted ``.dbn.zst``
            # files; legacy fakes return a store-like object directly.
            merged = _MergedDBNStore.from_download_result(raw)
            if isinstance(merged, _MergedDBNStore):
                # Materialize records + symbology while the source files
                # are still on disk; each DBN chunk is opened and closed
                # independently to avoid exhausting file handles.
                store: Any = self._materialize_downloaded_batch(merged, fallback_symbol=job.symbol)
            else:
                # Legacy fake — already in-memory; nothing to materialize.
                store = merged

        # Staging dir is gone. Now safe to drop the resume marker —
        # the records are entirely in memory.
        self._delete_marker(job.signature)
        return store

    @staticmethod
    def _timeframe_for_schema(schema: str) -> str:
        for timeframe, candidate in SCHEMA_BY_TIMEFRAME.items():
            if candidate == schema:
                return timeframe
        return schema

    def _materialize_downloaded_batch(
        self,
        merged: _MergedDBNStore,
        *,
        fallback_symbol: str,
    ) -> _MaterializedDBNStore:
        """Read downloaded DBN chunks without holding every file open."""
        import databento as db  # noqa: PLC0415 — import only when batch path runs

        records: list[DatabentoBarRecord] = []
        merged_mappings: dict[str, Any] = {}
        schema_name: str | None = None
        for path in merged.paths:
            store = db.DBNStore.from_file(str(path))
            try:
                if schema_name is None:
                    meta = getattr(store, "metadata", None)
                    raw_schema = getattr(store, "schema", None) or (
                        getattr(meta, "schema", None) if meta else None
                    )
                    schema_name = str(raw_schema) if raw_schema else None
                records.extend(self._extract_records(store, fallback_symbol=fallback_symbol))
                self._merge_batch_symbology(merged_mappings, store)
            finally:
                close = getattr(store, "close", None)
                if callable(close):
                    close()

        return _MaterializedDBNStore(
            records=tuple(records),
            symbology={"mappings": merged_mappings},
            schema_name=schema_name,
        )

    @staticmethod
    def _merge_batch_symbology(merged_mappings: dict[str, Any], store: Any) -> None:
        sym = getattr(store, "symbology", None) or {}
        if not isinstance(sym, dict):
            return
        mappings = sym.get("mappings")
        if isinstance(mappings, dict):
            merged_mappings.update(mappings)
            return
        for key, value in sym.items():
            if key != "mappings":
                merged_mappings.setdefault(key, value)

    def _extract_records(
        self,
        store: Any,
        *,
        fallback_symbol: str,
    ) -> list[DatabentoBarRecord]:
        """Pull our normalized record list out of a DBN-like response.

        Iteration order of preference:

        1. The store is directly iterable (real ``databento.DBNStore``
           supports ``for msg in store``; that's the documented happy
           path and avoids the push-style ``replay(callback)`` API).
        2. The store exposes a no-arg ``replay()`` returning a sequence
           — kept as a compatibility hook for fakes that predate the
           iterator-based contract.

        For real DBN messages we resolve the human-readable symbol via
        ``store.symbology`` (instrument id → raw symbol). Fakes can
        attach the symbol directly to each record; the loop short-
        circuits in that case.
        """
        symbol_map = self._build_instrument_to_symbol(store)
        raw_iter: Any
        if hasattr(store, "__iter__"):
            raw_iter = iter(store)
        else:
            replay_fn = getattr(store, "replay", None)
            raw_iter = iter(replay_fn()) if callable(replay_fn) else iter(())

        records: list[DatabentoBarRecord] = []
        for msg in raw_iter:
            if isinstance(msg, DatabentoBarRecord):
                symbol = (
                    fallback_symbol
                    if _same_databento_symbol(msg.symbol, fallback_symbol)
                    else msg.symbol
                )
                records.append(
                    DatabentoBarRecord(
                        ts_event_ns=msg.ts_event_ns,
                        instrument_id=msg.instrument_id,
                        open_q9=msg.open_q9,
                        high_q9=msg.high_q9,
                        low_q9=msg.low_q9,
                        close_q9=msg.close_q9,
                        volume=msg.volume,
                        symbol=symbol,
                    )
                )
                continue
            instrument_id = int(msg.instrument_id)
            symbol = symbol_map.get(instrument_id) or fallback_symbol
            if _same_databento_symbol(str(symbol), fallback_symbol):
                symbol = fallback_symbol
            records.append(
                DatabentoBarRecord(
                    ts_event_ns=int(msg.ts_event),
                    instrument_id=instrument_id,
                    open_q9=int(msg.open),
                    high_q9=int(msg.high),
                    low_q9=int(msg.low),
                    close_q9=int(msg.close),
                    volume=int(msg.volume),
                    symbol=str(symbol),
                )
            )
        return records

    @staticmethod
    def _build_instrument_to_symbol(store: Any) -> dict[int, str]:
        """Build inverse of the symbology table — instrument id → raw symbol.

        Reads ``store.symbology`` directly and reuses the shape-tolerant
        flattener in :func:`_symbology_to_dataframe` so we don't carry
        two copies of the shape-detection logic. Returns an empty dict
        if the store doesn't expose a symbology mapping; callers then
        fall back to the request symbol.
        """
        sym_obj = getattr(store, "symbology", None)
        if not isinstance(sym_obj, dict):
            return {}
        sym_df = _symbology_to_dataframe(sym_obj)
        if sym_df.is_empty():
            return {}
        return {
            int(row["instrument_id"]): str(row["raw_symbol"])
            for row in sym_df.iter_rows(named=True)
        }

    def _persist_symbology(self, store: Any) -> None:
        symbology = getattr(store, "symbology", None) or {}
        if not symbology:
            return
        df = _symbology_to_dataframe(symbology)
        if df.is_empty():
            return
        if self._store is None:
            # Compute-only; the runner is responsible for routing to a
            # store later if it wants persistence.
            return
        self._store.write(SYMBOLOGY_KEY, df, mode="append")

    def resolve_symbol_for_date(self, symbol: str, as_of: date) -> int | None:
        """Return the venue instrument id for ``symbol`` valid on ``as_of``.

        Reads the persisted symbology table written by
        :meth:`_persist_symbology` — one row per ``(raw_symbol,
        instrument_id)`` mapping with ``valid_from`` / ``valid_to``
        ISO-date strings. Returns ``None`` when no row covers
        ``as_of``, so callers can fail loud rather than silently picking
        the wrong id across a rename (e.g., ``FB → META``).
        """
        if self._store is None or not self._store.exists(SYMBOLOGY_KEY):
            return None
        df = self._store.read(SYMBOLOGY_KEY)
        if df.is_empty():
            return None
        iso = as_of.isoformat()
        valid_from = (
            pl.when(pl.col("valid_from").is_null() | (pl.col("valid_from") == ""))
            .then(pl.lit("0001-01-01"))
            .otherwise(pl.col("valid_from"))
        )
        valid_to = (
            pl.when(pl.col("valid_to").is_null() | (pl.col("valid_to") == ""))
            .then(pl.lit("9999-12-31"))
            .otherwise(pl.col("valid_to"))
        )
        matching = df.filter(
            (pl.col("raw_symbol") == symbol) & (valid_from <= iso) & (valid_to >= iso)
        )
        if matching.is_empty():
            return None
        return int(matching["instrument_id"][0])


__all__ = [
    "AGGREGATE_TOLERANCE",
    "AGGREGATION_WINDOWS",
    "AggregateCrossCheckError",
    "DATABENTO_BATCH_THRESHOLD_DAYS",
    "DATABENTO_DEFAULT_BACKOFF_BASE_SECONDS",
    "DATABENTO_DEFAULT_BATCH_JOBS_DIR",
    "DATABENTO_DEFAULT_BATCH_POLL_SECONDS",
    "DATABENTO_DEFAULT_MAX_RETRY_ATTEMPTS",
    "DATABENTO_PRICE_SCALE",
    "DATABENTO_UNDEF_PRICE",
    "DATASET_ROUTING",
    "SCHEMA_BY_TIMEFRAME",
    "SYMBOLOGY_KEY",
    "DatabentoBarRecord",
    "DatabentoError",
    "DatabentoProvider",
    "DatabentoRateLimitError",
    "DatabentoSchemaError",
    "DatabentoTransientError",
    "_estimate_bytes_in",
    "_records_to_dataframe",
    "_symbology_to_dataframe",
    "aggregate_bars",
]

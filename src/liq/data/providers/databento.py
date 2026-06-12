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
  a Databento dataset code via :data:`DATASET_ROUTING`. v1 ships equities
  only; other asset classes raise. The mapping is data, not control
  flow, so futures and crypto slot in later without protocol changes.
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
  ``event=databento_fetch`` and the structured fields listed in plan
  §3.2 (``provider, dataset, symbols_count, start, end, request_kind,
  bytes_in, duration_ms, sync_run_id``). The record uses ``extra=`` so
  the fields are queryable attributes on the ``LogRecord`` rather than
  baked into the message.
* **Network policy.** No real Databento call ever happens unless the
  caller supplies a real ``databento.Historical`` instance. Tests
  inject a fake client and never hit the network; the real-API smoke
  test is opt-in via ``RUN_DATABENTO=1`` (plan §1.5).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import polars as pl

from liq.data.exceptions import ProviderError
from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE, BaseProvider

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

SYMBOLOGY_KEY = "reference/databento/symbology"
"""``liq-store`` key prefix for the symbology mapping table."""

SCHEMA_BY_TIMEFRAME: dict[str, str] = {
    "1m": "ohlcv-1m",
    "1d": "ohlcv-1d",
}
"""``timeframe → Databento schema name``."""


# ----- records ---------------------------------------------------------------


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
        for r in records
    ]
    return pl.DataFrame(rows, schema=schema)


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
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for Databento provider")
        self._api_key = api_key
        self.asset_class = asset_class
        self.batch_threshold_days = batch_threshold_days
        self._client = client
        self._store = store

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
    ) -> pl.DataFrame:
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        dataset = self._resolve_dataset(timeframe)
        schema = self._resolve_schema(timeframe)

        span_days = (end - start).days + 1
        request_kind = "batch" if span_days >= self.batch_threshold_days else "get_range"

        start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=UTC)

        t0 = time.monotonic()
        store = self._invoke_remote(
            request_kind=request_kind,
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            start=start_dt,
            end=end_dt,
        )
        records = self._extract_records(store, fallback_symbol=symbol)
        df = _records_to_dataframe(records)
        self._persist_symbology(store)
        duration_ms = int((time.monotonic() - t0) * 1000)
        bytes_in = _estimate_bytes_in(store=store, rows=df.height)
        sync_run_id = str(uuid4())

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
                symbols=[symbol],
                schema=schema,
                start=start,
                end=end,
            )
        # Batch path. The current implementation routes the whole
        # window through one job and waits for completion before
        # downloading; partial-recovery + async polling are tracked as
        # follow-up hardening work (see the provider's docs for the
        # full hardening backlog).
        job = client.batch.submit_job(
            dataset=dataset,
            symbols=[symbol],
            schema=schema,
            start=start,
            end=end,
            encoding="dbn",
            stype_in="raw_symbol",
            stype_out="instrument_id",
        )
        return client.batch.download(job_id=job.get("id"))

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
                records.append(msg)
                continue
            instrument_id = int(msg.instrument_id)
            symbol = symbol_map.get(instrument_id) or fallback_symbol
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


__all__ = [
    "DATABENTO_BATCH_THRESHOLD_DAYS",
    "DATABENTO_PRICE_SCALE",
    "DATASET_ROUTING",
    "SCHEMA_BY_TIMEFRAME",
    "SYMBOLOGY_KEY",
    "DatabentoBarRecord",
    "DatabentoProvider",
    "_estimate_bytes_in",
    "_records_to_dataframe",
    "_symbology_to_dataframe",
]

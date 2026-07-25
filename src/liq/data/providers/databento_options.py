"""Databento OPRA options adapter (gamma-flow inputs).

A *sibling* to :class:`~liq.data.providers.databento.DatabentoProvider`: it
reuses that adapter's tested client + retry primitives but reads the
non-bar OPRA schemas needed for a dealer-gamma base signal —

* ``definition``  → the option chain (strike, expiry, put/call, multiplier),
* ``statistics``  → open interest (``stat_type == OPEN_INTEREST``),
* ``ohlcv-1d``    → daily prices per contract (for a causal mid).

It computes **no** greeks/IV (those are market features owned by
``liq-features``). Only ``definition``/``statistics``/``ohlcv-1d`` are ever
requested — never trades or tick/MBP — which bounds the OPRA billable size.
Cost is surfaced up-front via the free ``metadata.get_cost`` endpoint.

Reads are pay-as-you-go historical (no live subscription). Fold governance
is enforced separately by the lockbox guard via the ``databento_opra_options``
dataset (``asset_class="option"``); this adapter only loads/normalizes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import polars as pl

from liq.data.options import (
    CHAIN_CONTRACTS_SCHEMA,
    GAMMA_FLOW_ROWS_SCHEMA,
    GammaFlowFrame,
    OptionChainSnapshot,
    empty_chain_contracts,
)
from liq.data.providers.base import PRICE_DTYPE
from liq.data.providers.databento import _q9_to_decimal

__all__ = [
    "OPRA_DATASET",
    "OPRA_SCHEMAS",
    "DatabentoOptionsAdapter",
]

OPRA_DATASET = "OPRA.PILLAR"
#: Only these three schemas are ever requested (bounds billable size).
OPRA_SCHEMAS = {
    "definition": "definition",
    "open_interest": "statistics",
    "ohlcv": "ohlcv-1d",
}
_OPEN_INTEREST_STAT_TYPE = 9  # databento_dbn.StatType.OPEN_INTEREST

#: Cash-settled index underlyings carry European-exercise options; everything
#: else (ETFs, single names) is American. Used only to tag the chain so the
#: greek model downstream can pick the right pricer. Frozen rule.
_EUROPEAN_INDEX_UNDERLYINGS = frozenset({"SPX", "SPXW", "NDX", "NDXP", "RUT", "VIX"})


def _parent_symbol(underlying: str) -> str:
    """OPRA parent-symbology key for one underlying (all its listed options)."""
    return f"{underlying.upper()}.OPT"


def _ns_to_date(ns: int) -> date:
    """Convert a Databento UNIX-nanosecond timestamp to a UTC calendar date."""
    return datetime.fromtimestamp(ns // 1_000_000_000, tz=UTC).date()


def _exercise_style(underlying: str) -> str:
    return "E" if underlying.upper() in _EUROPEAN_INDEX_UNDERLYINGS else "A"


class DatabentoOptionsAdapter:
    """Loads + normalizes OPRA chain/OI/price data for a gamma-flow signal."""

    def __init__(
        self,
        api_key: str,
        *,
        client: Any | None = None,
        store: Any | None = None,
        **base_kwargs: Any,
    ) -> None:
        # Compose the bars provider purely to reuse its tested lazy client and
        # bounded-retry primitives; we never call its bar methods.
        from liq.data.providers.databento import DatabentoProvider  # noqa: PLC0415

        self._base = DatabentoProvider(
            api_key, asset_class="option", client=client, store=store, **base_kwargs
        )
        self._store = store

    @property
    def name(self) -> str:
        return "databento_options"

    # -- remote plumbing (reuses the bars adapter's client + retry) ----------

    def _get_range(self, *, underlying: str, schema: str, start: date, end: date) -> Any:
        client = self._base._get_client()

        def op() -> Any:
            return client.timeseries.get_range(
                dataset=OPRA_DATASET,
                symbols=[_parent_symbol(underlying)],
                stype_in="parent",
                schema=schema,
                start=datetime(start.year, start.month, start.day, tzinfo=UTC),
                end=datetime(end.year, end.month, end.day, tzinfo=UTC),
            )

        return self._base._call_with_retry(
            op,
            sync_run_id=f"opra:{schema}:{underlying}",
            dataset=OPRA_DATASET,
            symbol=_parent_symbol(underlying),
            request_kind="get_range",
        )

    def estimate_cost(
        self, *, underlying: str, schema: str, start: date, end: date
    ) -> dict[str, Any]:
        """Free ``get_cost``/billable-size estimate for one OPRA query (no download)."""
        client = self._base._get_client()
        params = {
            "dataset": OPRA_DATASET,
            "symbols": [_parent_symbol(underlying)],
            "stype_in": "parent",
            "schema": schema,
            "start": datetime(start.year, start.month, start.day, tzinfo=UTC),
            "end": datetime(end.year, end.month, end.day, tzinfo=UTC),
        }
        return {
            "underlying": underlying,
            "schema": schema,
            "cost_usd": client.metadata.get_cost(**params),
            "billable_bytes": client.metadata.get_billable_size(**params),
        }

    # -- normalizers (duck-typed on the databento record attributes) ---------

    @staticmethod
    def _definitions_to_chain(records: Any, underlying: str) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        style = _exercise_style(underlying)
        for rec in records:
            option_type = getattr(rec, "instrument_class", None)
            if option_type not in ("C", "P"):
                continue  # skip non-option / spread / underlying rows
            rows.append(
                {
                    "instrument_id": int(rec.instrument_id),
                    "osi_symbol": str(rec.raw_symbol),
                    "option_type": option_type,
                    "strike": _q9_to_decimal(int(rec.strike_price)),
                    "expiration": _ns_to_date(int(rec.expiration)),
                    "contract_multiplier": int(getattr(rec, "contract_multiplier", 0) or 0),
                    "exercise_style": style,
                }
            )
        if not rows:
            return empty_chain_contracts()
        return pl.DataFrame(rows, schema=CHAIN_CONTRACTS_SCHEMA)

    @staticmethod
    def _statistics_to_oi(records: Any) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for rec in records:
            if int(getattr(rec, "stat_type", -1)) != _OPEN_INTEREST_STAT_TYPE:
                continue
            ref = int(getattr(rec, "ts_ref", 0) or getattr(rec, "ts_event", 0))
            rows.append(
                {
                    "instrument_id": int(rec.instrument_id),
                    "oi_as_of": _ns_to_date(ref),
                    "open_interest": int(rec.quantity),
                }
            )
        schema = {
            "instrument_id": pl.UInt32(),
            "oi_as_of": pl.Date(),
            "open_interest": pl.Int64(),
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)

    @staticmethod
    def _ohlcv_to_prices(records: Any) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for rec in records:
            high = _q9_to_decimal(int(rec.high))
            low = _q9_to_decimal(int(rec.low))
            rows.append(
                {
                    "instrument_id": int(rec.instrument_id),
                    "session": _ns_to_date(int(rec.ts_event)),
                    "mid": (high + low) / 2,
                }
            )
        schema = {
            "instrument_id": pl.UInt32(),
            "session": pl.Date(),
            "mid": PRICE_DTYPE,
        }
        if not rows:
            return pl.DataFrame(schema=schema)
        return pl.DataFrame(rows, schema=schema)

    # -- public port surface -------------------------------------------------

    def fetch_chain_snapshot(self, underlying: str, as_of: date) -> OptionChainSnapshot:
        records = self._get_range(
            underlying=underlying, schema=OPRA_SCHEMAS["definition"], start=as_of, end=as_of
        )
        contracts = self._definitions_to_chain(records, underlying)
        return OptionChainSnapshot(underlying=underlying, as_of=as_of, contracts=contracts)

    def fetch_open_interest(self, underlying: str, start: date, end: date) -> pl.DataFrame:
        records = self._get_range(
            underlying=underlying, schema=OPRA_SCHEMAS["open_interest"], start=start, end=end
        )
        return self._statistics_to_oi(records)

    def fetch_option_ohlcv(self, underlying: str, start: date, end: date) -> pl.DataFrame:
        records = self._get_range(
            underlying=underlying, schema=OPRA_SCHEMAS["ohlcv"], start=start, end=end
        )
        return self._ohlcv_to_prices(records)

    def build_gamma_flow_frame(
        self,
        underlying: str,
        as_of: date,
        *,
        oi_lag_sessions: int,
        spot: Decimal,
        spot_as_of: date,
    ) -> GammaFlowFrame:
        """Join chain + causally-lagged OI + daily mid + spot into one frame.

        Open interest for session ``as_of`` is only *known* the next session, so
        the OI used is the latest published on or before ``as_of`` shifted back
        ``oi_lag_sessions`` calendar days; ``feature_available_at`` is set one
        day past ``as_of`` accordingly. The caller supplies the underlying
        ``spot``/``spot_as_of`` (equity data lives in a different dataset).
        Rows missing OI or a mid are dropped — never imputed.
        """
        if oi_lag_sessions < 1:
            raise ValueError("oi_lag_sessions must be >= 1 (open interest settles T+1)")
        if spot_as_of > as_of:
            raise ValueError("spot_as_of must be on or before as_of (no look-ahead)")

        chain = self.fetch_chain_snapshot(underlying, as_of).contracts
        oi_cutoff = as_of - timedelta(days=oi_lag_sessions)
        oi = (
            self.fetch_open_interest(underlying, oi_cutoff - timedelta(days=7), oi_cutoff)
            .sort("oi_as_of")
            .group_by("instrument_id")
            .last()  # latest OI known by the lagged cutoff
        )
        prices = self.fetch_option_ohlcv(underlying, as_of, as_of).select("instrument_id", "mid")

        rows = (
            chain.join(oi, on="instrument_id", how="inner")
            .join(prices, on="instrument_id", how="inner")
            .with_columns(
                ((pl.col("expiration") - pl.lit(as_of)).dt.total_days() / 365.25).alias(
                    "tte_years"
                ),
                pl.lit(spot, dtype=PRICE_DTYPE).alias("underlying_spot"),
                pl.lit(spot_as_of).alias("spot_as_of"),
            )
            .filter(pl.col("tte_years") > 0.0)
            .select(list(GAMMA_FLOW_ROWS_SCHEMA.keys()))
        )
        available = datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC) + timedelta(days=1)
        return GammaFlowFrame(
            underlying=underlying,
            as_of=as_of,
            feature_available_at=available,
            rows=rows,
        )

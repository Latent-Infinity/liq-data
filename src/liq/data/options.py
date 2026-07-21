"""Domain port and types for options / dealer-gamma inputs.

This module defines the *contract* a data adapter must satisfy to feed a
gamma-flow (dealer-gamma / GEX) base signal: normalized option-chain
definitions, open interest, and daily option prices for one underlying.

It is deliberately vendor-agnostic and computes no greeks/IV — those are
market features owned by ``liq-features``. The port keeps ``liq-data``
responsible only for *loading and normalizing* vendor data (chain, OI,
prices), with a causal ``feature_available_at`` so downstream code cannot
peek at data before it was knowable.

Prices/strikes are exact ``Decimal`` (never float round-tripped); open
interest is an integer count; missing rows are excluded upstream, never
imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import polars as pl

from liq.data.providers.base import PRICE_DTYPE

__all__ = [
    "CHAIN_CONTRACTS_SCHEMA",
    "GAMMA_FLOW_ROWS_SCHEMA",
    "GammaFlowDataPort",
    "GammaFlowFrame",
    "OptionChainSnapshot",
    "empty_chain_contracts",
    "empty_gamma_flow_rows",
]

# --- normalized schemas -------------------------------------------------------

#: Chain definition rows (one per listed contract for an underlying/session).
CHAIN_CONTRACTS_SCHEMA: dict[str, pl.DataType] = {
    "instrument_id": pl.UInt32(),
    "osi_symbol": pl.Utf8(),
    "option_type": pl.Utf8(),  # "C" | "P"
    "strike": PRICE_DTYPE,
    "expiration": pl.Date(),
    "contract_multiplier": pl.Int32(),
    "exercise_style": pl.Utf8(),  # "E" (European) | "A" (American)
}

#: Per-underlying, per-decision-date normalized gamma-flow input rows.
#: Vendor-normalized only — greeks/IV are filled by liq-features, not here.
GAMMA_FLOW_ROWS_SCHEMA: dict[str, pl.DataType] = {
    "instrument_id": pl.UInt32(),
    "option_type": pl.Utf8(),
    "strike": PRICE_DTYPE,
    "expiration": pl.Date(),
    "tte_years": pl.Float64(),
    "open_interest": pl.Int64(),
    "oi_as_of": pl.Date(),
    "mid": PRICE_DTYPE,
    "underlying_spot": PRICE_DTYPE,
    "spot_as_of": pl.Date(),
    "contract_multiplier": pl.Int32(),
    "exercise_style": pl.Utf8(),
}


def empty_chain_contracts() -> pl.DataFrame:
    """Typed-empty chain-contracts frame (no imputation for missing chains)."""
    return pl.DataFrame(schema=CHAIN_CONTRACTS_SCHEMA)


def empty_gamma_flow_rows() -> pl.DataFrame:
    """Typed-empty gamma-flow rows frame."""
    return pl.DataFrame(schema=GAMMA_FLOW_ROWS_SCHEMA)


# --- domain types -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptionChainSnapshot:
    """Chain definition for one underlying as-of one session.

    ``contracts`` conforms to :data:`CHAIN_CONTRACTS_SCHEMA`.
    """

    underlying: str
    as_of: date
    contracts: pl.DataFrame

    def __post_init__(self) -> None:
        _require_schema(self.contracts, CHAIN_CONTRACTS_SCHEMA, "OptionChainSnapshot.contracts")


@dataclass(frozen=True, slots=True)
class GammaFlowFrame:
    """Per-underlying, per-decision-date normalized options input frame.

    ``rows`` conforms to :data:`GAMMA_FLOW_ROWS_SCHEMA`. ``feature_available_at``
    (UTC) declares when this frame is usable for a decision; because option
    open interest settles overnight it must be strictly after ``as_of``.
    """

    underlying: str
    as_of: date
    feature_available_at: datetime
    rows: pl.DataFrame

    def __post_init__(self) -> None:
        _require_schema(self.rows, GAMMA_FLOW_ROWS_SCHEMA, "GammaFlowFrame.rows")
        if self.feature_available_at.tzinfo is None:
            raise ValueError("feature_available_at must be timezone-aware (UTC)")
        if self.feature_available_at.date() <= self.as_of:
            # Open interest is a T+1 quantity; a frame usable on or before its
            # own session date would be look-ahead.
            raise ValueError(
                "feature_available_at must be strictly after as_of "
                f"(got {self.feature_available_at.date()} <= {self.as_of})"
            )


# --- port ---------------------------------------------------------------------


@runtime_checkable
class GammaFlowDataPort(Protocol):
    """Contract an adapter satisfies to feed a dealer-gamma base signal."""

    @property
    def name(self) -> str: ...

    def fetch_chain_snapshot(self, underlying: str, as_of: date) -> OptionChainSnapshot: ...

    def fetch_open_interest(self, underlying: str, start: date, end: date) -> pl.DataFrame: ...

    def fetch_option_ohlcv(self, underlying: str, start: date, end: date) -> pl.DataFrame: ...

    def build_gamma_flow_frame(
        self, underlying: str, as_of: date, *, oi_lag_sessions: int
    ) -> GammaFlowFrame: ...


def _require_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType], label: str) -> None:
    expected = list(schema.keys())
    if frame.columns != expected:
        raise ValueError(f"{label} columns {frame.columns} != expected {expected}")
    for col, dtype in schema.items():
        if frame.schema[col] != dtype:
            raise ValueError(f"{label} column '{col}' dtype {frame.schema[col]} != {dtype}")

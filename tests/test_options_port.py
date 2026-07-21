"""Contract tests for the gamma-flow options domain port.

Uses an in-process fake adapter (no network) so the suite is deterministic.
Exercises schema/dtype conformance, Decimal-exact strikes, the causal
``feature_available_at`` rule, structural (runtime_checkable) conformance,
and typed-empty handling.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from liq.data.options import (
    CHAIN_CONTRACTS_SCHEMA,
    GAMMA_FLOW_ROWS_SCHEMA,
    GammaFlowDataPort,
    GammaFlowFrame,
    OptionChainSnapshot,
    empty_chain_contracts,
    empty_gamma_flow_rows,
)
from liq.data.providers.base import PRICE_DTYPE


class _FakeOptionsAdapter:
    """Minimal in-process GammaFlowDataPort for contract testing."""

    @property
    def name(self) -> str:
        return "fake_options"

    def fetch_chain_snapshot(self, underlying: str, as_of: date) -> OptionChainSnapshot:
        contracts = pl.DataFrame(
            {
                "instrument_id": [1, 2],
                "osi_symbol": [f"{underlying}  C", f"{underlying}  P"],
                "option_type": ["C", "P"],
                "strike": [Decimal("100.00000000"), Decimal("100.00000000")],
                "expiration": [date(2021, 1, 15), date(2021, 1, 15)],
                "contract_multiplier": [100, 100],
                "exercise_style": ["A", "A"],
            },
            schema=CHAIN_CONTRACTS_SCHEMA,
        )
        return OptionChainSnapshot(underlying=underlying, as_of=as_of, contracts=contracts)

    def fetch_open_interest(self, underlying: str, start: date, end: date) -> pl.DataFrame:
        return pl.DataFrame(
            {"instrument_id": [1], "oi_as_of": [start], "open_interest": [1234]},
            schema={
                "instrument_id": pl.UInt32(),
                "oi_as_of": pl.Date(),
                "open_interest": pl.Int64(),
            },
        )

    def fetch_option_ohlcv(self, underlying: str, start: date, end: date) -> pl.DataFrame:
        return pl.DataFrame(
            {"instrument_id": [1], "session": [start], "mid": [Decimal("2.50000000")]},
            schema={
                "instrument_id": pl.UInt32(),
                "session": pl.Date(),
                "mid": PRICE_DTYPE,
            },
        )

    def build_gamma_flow_frame(
        self, underlying: str, as_of: date, *, oi_lag_sessions: int
    ) -> GammaFlowFrame:
        rows = pl.DataFrame(
            {
                "instrument_id": [1],
                "option_type": ["C"],
                "strike": [Decimal("100.00000000")],
                "expiration": [date(2021, 1, 15)],
                "tte_years": [0.05],
                "open_interest": [1234],
                "oi_as_of": [as_of],
                "mid": [Decimal("2.50000000")],
                "underlying_spot": [Decimal("101.25000000")],
                "spot_as_of": [as_of],
                "contract_multiplier": [100],
                "exercise_style": ["A"],
            },
            schema=GAMMA_FLOW_ROWS_SCHEMA,
        )
        # OI settles overnight: usable T+max(1, oi_lag_sessions) sessions later.
        available = datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC) + timedelta(
            days=max(1, oi_lag_sessions)
        )
        return GammaFlowFrame(
            underlying=underlying,
            as_of=as_of,
            feature_available_at=available,
            rows=rows,
        )


@pytest.fixture
def adapter() -> _FakeOptionsAdapter:
    return _FakeOptionsAdapter()


def test_adapter_is_structurally_a_gamma_flow_data_port(adapter: _FakeOptionsAdapter) -> None:
    assert isinstance(adapter, GammaFlowDataPort)


def test_chain_snapshot_schema_and_decimal_strikes(adapter: _FakeOptionsAdapter) -> None:
    snap = adapter.fetch_chain_snapshot("SPY", date(2021, 1, 4))
    assert snap.contracts.columns == list(CHAIN_CONTRACTS_SCHEMA)
    assert snap.contracts.schema["strike"] == PRICE_DTYPE
    # exact Decimal, not float
    assert snap.contracts["strike"][0] == Decimal("100.00000000")


def test_gamma_flow_frame_schema_and_causality(adapter: _FakeOptionsAdapter) -> None:
    frame = adapter.build_gamma_flow_frame("SPY", date(2021, 1, 4), oi_lag_sessions=1)
    assert frame.rows.columns == list(GAMMA_FLOW_ROWS_SCHEMA)
    assert frame.feature_available_at.tzinfo is not None
    # OI settles overnight → usable strictly after the session date.
    assert frame.feature_available_at.date() > frame.as_of


def test_frame_rejects_non_causal_availability() -> None:
    with pytest.raises(ValueError, match="strictly after as_of"):
        GammaFlowFrame(
            underlying="SPY",
            as_of=date(2021, 1, 4),
            feature_available_at=datetime(2021, 1, 4, 23, 0, tzinfo=UTC),
            rows=empty_gamma_flow_rows(),
        )


def test_frame_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        GammaFlowFrame(
            underlying="SPY",
            as_of=date(2021, 1, 4),
            feature_available_at=datetime(2021, 1, 5, 14, 30),
            rows=empty_gamma_flow_rows(),
        )


def test_snapshot_rejects_wrong_schema() -> None:
    with pytest.raises(ValueError, match="columns"):
        OptionChainSnapshot(
            underlying="SPY",
            as_of=date(2021, 1, 4),
            contracts=pl.DataFrame({"wrong": [1]}),
        )


def test_typed_empty_frames_conform() -> None:
    assert empty_chain_contracts().columns == list(CHAIN_CONTRACTS_SCHEMA)
    assert empty_gamma_flow_rows().columns == list(GAMMA_FLOW_ROWS_SCHEMA)
    # A snapshot/frame built from typed-empty frames is valid.
    OptionChainSnapshot("SPY", date(2021, 1, 4), empty_chain_contracts())

"""Contract tests for the Databento OPRA options adapter.

Uses an in-process fake databento client (no network, mirroring the bars
adapter's test policy) whose ``timeseries.get_range`` returns synthetic
records carrying the *real* OPRA field names, and whose ``metadata`` returns
canned cost figures. Validates normalization, the causal chain/OI/price join,
cost surfacing, parent symbology, and that only the three cheap schemas
(definition / statistics / ohlcv-1d) are ever requested.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from liq.data.options import GAMMA_FLOW_ROWS_SCHEMA, GammaFlowDataPort
from liq.data.providers.databento_options import OPRA_DATASET, DatabentoOptionsAdapter

Q9 = 1_000_000_000


def _ns(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) * Q9


def _definition_records() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            instrument_id=1,
            raw_symbol="SPY   210115C00100000",
            instrument_class="C",
            strike_price=100 * Q9,
            expiration=_ns(date(2021, 1, 15)),
            contract_multiplier=100,
        ),
        SimpleNamespace(
            instrument_id=2,
            raw_symbol="SPY   210115P00100000",
            instrument_class="P",
            strike_price=100 * Q9,
            expiration=_ns(date(2021, 1, 15)),
            contract_multiplier=100,
        ),
        # A non-option row (e.g. a spread/underlying) that must be dropped.
        SimpleNamespace(
            instrument_id=99,
            raw_symbol="SPY-SPREAD",
            instrument_class="M",
            strike_price=0,
            expiration=_ns(date(2021, 1, 15)),
            contract_multiplier=100,
        ),
    ]


def _statistics_records() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(instrument_id=1, stat_type=9, quantity=1234, ts_ref=_ns(date(2021, 1, 3))),
        SimpleNamespace(instrument_id=2, stat_type=9, quantity=5678, ts_ref=_ns(date(2021, 1, 3))),
        # A non-OI statistic (e.g. settlement price) that must be filtered out.
        SimpleNamespace(instrument_id=1, stat_type=3, quantity=0, ts_ref=_ns(date(2021, 1, 3))),
    ]


def _ohlcv_records() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            instrument_id=1,
            high=260 * Q9 // 100,
            low=240 * Q9 // 100,
            ts_event=_ns(date(2021, 1, 4)),
        ),
        SimpleNamespace(
            instrument_id=2,
            high=160 * Q9 // 100,
            low=140 * Q9 // 100,
            ts_event=_ns(date(2021, 1, 4)),
        ),
    ]


class _FakeMetadata:
    def get_cost(self, **kwargs: Any) -> float:
        return 0.0123

    def get_billable_size(self, **kwargs: Any) -> int:
        return 4096


class _FakeTimeseries:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_range(self, **kwargs: Any) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        schema = kwargs["schema"]
        if schema == "definition":
            return _definition_records()
        if schema == "statistics":
            return _statistics_records()
        if schema == "ohlcv-1d":
            return _ohlcv_records()
        raise AssertionError(f"unexpected schema requested: {schema}")


class _FakeClient:
    def __init__(self) -> None:
        self.timeseries = _FakeTimeseries()
        self.metadata = _FakeMetadata()


@pytest.fixture
def adapter() -> DatabentoOptionsAdapter:
    return DatabentoOptionsAdapter("db-fake", client=_FakeClient())


def test_adapter_is_a_gamma_flow_data_port(adapter: DatabentoOptionsAdapter) -> None:
    assert isinstance(adapter, GammaFlowDataPort)
    assert adapter.name == "databento_options"


def test_chain_snapshot_drops_non_options_and_decimalizes_strike(
    adapter: DatabentoOptionsAdapter,
) -> None:
    snap = adapter.fetch_chain_snapshot("SPY", date(2021, 1, 4))
    assert snap.contracts.height == 2  # the "M" spread row is dropped
    assert set(snap.contracts["option_type"]) == {"C", "P"}
    assert snap.contracts.filter(option_type="C")["strike"][0] == Decimal("100.00000000")
    assert snap.contracts["exercise_style"][0] == "A"  # SPY is American


def test_index_underlying_tagged_european() -> None:
    adapter = DatabentoOptionsAdapter("db-fake", client=_FakeClient())
    snap = adapter.fetch_chain_snapshot("SPX", date(2021, 1, 4))
    assert snap.contracts["exercise_style"][0] == "E"


def test_open_interest_filters_to_oi_stat_type(adapter: DatabentoOptionsAdapter) -> None:
    oi = adapter.fetch_open_interest("SPY", date(2020, 12, 28), date(2021, 1, 3))
    assert oi.height == 2  # the stat_type=3 row is filtered out
    assert oi.filter(instrument_id=1)["open_interest"][0] == 1234


def test_option_ohlcv_mid_is_hl_average(adapter: DatabentoOptionsAdapter) -> None:
    prices = adapter.fetch_option_ohlcv("SPY", date(2021, 1, 4), date(2021, 1, 4))
    assert prices.filter(instrument_id=1)["mid"][0] == Decimal("2.50000000")


def test_build_gamma_flow_frame_joins_causally(adapter: DatabentoOptionsAdapter) -> None:
    frame = adapter.build_gamma_flow_frame(
        "SPY",
        date(2021, 1, 4),
        oi_lag_sessions=1,
        spot=Decimal("372.50000000"),
        spot_as_of=date(2021, 1, 4),
    )
    assert frame.rows.columns == list(GAMMA_FLOW_ROWS_SCHEMA)
    assert frame.rows.height == 2
    assert frame.feature_available_at.date() > frame.as_of
    assert (frame.rows["tte_years"] > 0).all()
    assert frame.rows["underlying_spot"][0] == Decimal("372.50000000")
    assert frame.rows.filter(instrument_id=1)["open_interest"][0] == 1234


def test_build_frame_rejects_bad_oi_lag(adapter: DatabentoOptionsAdapter) -> None:
    with pytest.raises(ValueError, match="oi_lag_sessions"):
        adapter.build_gamma_flow_frame(
            "SPY",
            date(2021, 1, 4),
            oi_lag_sessions=0,
            spot=Decimal("1"),
            spot_as_of=date(2021, 1, 4),
        )


def test_build_frame_rejects_future_spot(adapter: DatabentoOptionsAdapter) -> None:
    with pytest.raises(ValueError, match="spot_as_of"):
        adapter.build_gamma_flow_frame(
            "SPY",
            date(2021, 1, 4),
            oi_lag_sessions=1,
            spot=Decimal("1"),
            spot_as_of=date(2021, 1, 5),
        )


def test_estimate_cost_uses_metadata(adapter: DatabentoOptionsAdapter) -> None:
    est = adapter.estimate_cost(
        underlying="SPY", schema="definition", start=date(2021, 1, 4), end=date(2021, 1, 4)
    )
    assert est["cost_usd"] == 0.0123
    assert est["billable_bytes"] == 4096


def test_only_cheap_schemas_and_parent_symbology_requested(
    adapter: DatabentoOptionsAdapter,
) -> None:
    adapter.build_gamma_flow_frame(
        "SPY", date(2021, 1, 4), oi_lag_sessions=1, spot=Decimal("1"), spot_as_of=date(2021, 1, 4)
    )
    calls = adapter._base._client.timeseries.calls  # type: ignore[attr-defined]
    schemas = {c["schema"] for c in calls}
    assert schemas <= {"definition", "statistics", "ohlcv-1d"}
    assert "trades" not in schemas and "mbp-1" not in schemas
    for c in calls:
        assert c["dataset"] == OPRA_DATASET
        assert c["symbols"] == ["SPY.OPT"]
        assert c["stype_in"] == "parent"

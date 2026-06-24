"""Tests for DataService.estimate_databento_cost.

The method provides a no-spend pre-fetch cost gate for Databento ingestion.
It wraps Databento's non-billable metadata endpoints (`metadata.get_cost`
and `metadata.get_billable_size`) so a B1-style caller can confirm an
operator-authorised cost bound before invoking the billable
`DataService.sync(...)`.

Test isolation: every test injects a fake Databento `Historical` class via
monkeypatch so no network calls happen.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from liq.data import DataService
from liq.data.settings import LiqDataSettings
from liq.data.universes import UniverseDefinition, UniverseKind


class _RecordingMetadata:
    """Record every call so tests can assert the exact arg list."""

    def __init__(self, cost_usd: float, billable_bytes: int) -> None:
        self._cost_usd = cost_usd
        self._billable_bytes = billable_bytes
        self.get_cost_calls: list[dict[str, Any]] = []
        self.get_billable_size_calls: list[dict[str, Any]] = []

    def get_cost(self, **kwargs: Any) -> float:
        self.get_cost_calls.append(dict(kwargs))
        return self._cost_usd

    def get_billable_size(self, **kwargs: Any) -> int:
        self.get_billable_size_calls.append(dict(kwargs))
        return self._billable_bytes


class _FakeHistorical:
    """Stand-in for `databento.Historical`. Records the constructor call."""

    init_calls: list[dict[str, Any]] = []
    metadata_instance: _RecordingMetadata | None = None
    timeseries_calls: int = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self).init_calls.append(dict(kwargs))
        if type(self).metadata_instance is None:
            type(self).metadata_instance = _RecordingMetadata(
                cost_usd=2.50, billable_bytes=11_500_000
            )
        self.metadata = type(self).metadata_instance

    @property
    def timeseries(self) -> Any:
        type(self).timeseries_calls += 1
        raise AssertionError("timeseries access is billable; estimate must not touch it")


@pytest.fixture(autouse=True)
def _reset_fake_historical() -> None:
    _FakeHistorical.init_calls = []
    _FakeHistorical.metadata_instance = None
    _FakeHistorical.timeseries_calls = 0


@pytest.fixture
def fake_databento_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject a fake `databento` module exposing only `Historical`."""
    import sys
    import types

    fake_module = types.ModuleType("databento")
    fake_module.Historical = _FakeHistorical  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "databento", fake_module)
    return fake_module


@pytest.fixture
def service_with_api_key(tmp_path: Path) -> DataService:
    """DataService configured with a fake api_key (the value matters only for the call assertion)."""
    settings = LiqDataSettings(databento_api_key="db-test-key-123", data_root=tmp_path)
    return DataService(settings=settings, data_root=tmp_path)


@pytest.fixture
def aichip_universe() -> UniverseDefinition:
    """A 4-symbol slice of ai-chip-liquid-30-riskvar (subset is sufficient for unit-test isolation)."""
    return UniverseDefinition(
        name="aichip-slice",
        version=1,
        kind=UniverseKind.EXPLICIT,
        spec={"symbols": ["NVDA", "AAPL", "MSFT", "AMD"]},
    )


# ---------------------------------------------------------------------- #
# happy path                                                              #
# ---------------------------------------------------------------------- #


class TestEstimateDatabentoCostHappyPath:
    """Happy-path coverage of the new method."""

    def test_returns_dict_with_required_keys(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )

        assert isinstance(result, dict)
        for key in (
            "billable_bytes",
            "estimated_cost_usd",
            "dataset",
            "schema",
            "symbols",
            "start",
            "end",
            "provider_request_id",
        ):
            assert key in result, f"missing key: {key}"

    def test_derives_schema_from_timeframe(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert result["schema"] == "ohlcv-1m"

    def test_calls_metadata_get_cost_with_derived_schema(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        metadata = _FakeHistorical.metadata_instance
        assert metadata is not None
        assert len(metadata.get_cost_calls) == 1
        call = metadata.get_cost_calls[0]
        assert call["dataset"] == "EQUS.MINI"
        assert call["schema"] == "ohlcv-1m"
        assert call["start"] == date(2021, 1, 4)
        assert call["end"] == date(2021, 1, 8)
        assert sorted(call["symbols"]) == ["AAPL", "AMD", "MSFT", "NVDA"]

    def test_calls_metadata_get_billable_size_with_derived_schema(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        metadata = _FakeHistorical.metadata_instance
        assert metadata is not None
        assert len(metadata.get_billable_size_calls) == 1
        call = metadata.get_billable_size_calls[0]
        assert call["dataset"] == "EQUS.MINI"
        assert call["schema"] == "ohlcv-1m"

    def test_constructor_uses_key_not_api_key(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        """databento 0.79.0 Historical constructor takes key=..., not api_key=..."""
        service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert len(_FakeHistorical.init_calls) == 1
        init_kwargs = _FakeHistorical.init_calls[0]
        assert "key" in init_kwargs
        assert init_kwargs["key"] == "db-test-key-123"
        assert "api_key" not in init_kwargs

    def test_returned_cost_is_get_cost_return_value(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert result["estimated_cost_usd"] == pytest.approx(2.50)

    def test_returned_bytes_is_get_billable_size_return_value(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert result["billable_bytes"] == 11_500_000

    def test_symbols_list_passed_through(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert sorted(result["symbols"]) == ["AAPL", "AMD", "MSFT", "NVDA"]

    def test_provider_request_id_is_non_empty_string(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert isinstance(result["provider_request_id"], str)
        assert len(result["provider_request_id"]) > 0

    def test_no_billable_endpoint_touched(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        """timeseries access would be billable; estimate must never touch it."""
        service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert _FakeHistorical.timeseries_calls == 0


# ---------------------------------------------------------------------- #
# rejection paths                                                         #
# ---------------------------------------------------------------------- #


class TestEstimateDatabentoCostRejection:
    """Rejection paths for invalid inputs."""

    @pytest.mark.parametrize("bad_timeframe", ["5m", "1h", "1d", "30s", ""])
    def test_rejects_non_1m_timeframe(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
        bad_timeframe: str,
    ) -> None:
        """v0.2.6 contract: only timeframe='1m' is supported."""
        with pytest.raises(ValueError, match="timeframe"):
            service_with_api_key.estimate_databento_cost(
                aichip_universe,
                start=date(2021, 1, 4),
                end=date(2021, 1, 8),
                timeframe=bad_timeframe,
                dataset="EQUS.MINI",
            )

    def test_rejects_missing_api_key(
        self,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
        tmp_path: Path,
    ) -> None:
        """Without DATABENTO_API_KEY, estimate must refuse."""
        settings = LiqDataSettings(databento_api_key=None, data_root=tmp_path)
        ds = DataService(settings=settings, data_root=tmp_path)
        with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
            ds.estimate_databento_cost(
                aichip_universe,
                start=date(2021, 1, 4),
                end=date(2021, 1, 8),
                timeframe="1m",
                dataset="EQUS.MINI",
            )

    def test_rejects_empty_universe(
        self,
        service_with_api_key: DataService,
        fake_databento_module: Any,
    ) -> None:
        empty = UniverseDefinition(
            name="empty",
            version=1,
            kind=UniverseKind.EXPLICIT,
            spec={"symbols": []},
        )
        with pytest.raises(ValueError, match="symbols"):
            service_with_api_key.estimate_databento_cost(
                empty,
                start=date(2021, 1, 4),
                end=date(2021, 1, 8),
                timeframe="1m",
                dataset="EQUS.MINI",
            )

    def test_rejects_end_before_start(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        with pytest.raises(ValueError, match="start"):
            service_with_api_key.estimate_databento_cost(
                aichip_universe,
                start=date(2021, 1, 8),
                end=date(2021, 1, 4),
                timeframe="1m",
                dataset="EQUS.MINI",
            )


# ---------------------------------------------------------------------- #
# universe-ref shapes                                                     #
# ---------------------------------------------------------------------- #


class TestEstimateDatabentoCostUniverseShapes:
    """The universe-ref arg accepts the same shapes as DataService.resolve_universe."""

    def test_accepts_explicit_definition(
        self,
        service_with_api_key: DataService,
        aichip_universe: UniverseDefinition,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            aichip_universe,
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert sorted(result["symbols"]) == ["AAPL", "AMD", "MSFT", "NVDA"]

    def test_accepts_symbol_list(
        self,
        service_with_api_key: DataService,
        fake_databento_module: Any,
    ) -> None:
        result = service_with_api_key.estimate_databento_cost(
            ["NVDA", "AAPL"],
            start=date(2021, 1, 4),
            end=date(2021, 1, 8),
            timeframe="1m",
            dataset="EQUS.MINI",
        )
        assert sorted(result["symbols"]) == ["AAPL", "NVDA"]

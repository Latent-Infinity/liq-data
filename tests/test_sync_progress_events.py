"""TDD tests for sync progress events.

Operators running a full SP500 sync want to see the run is alive
between universe resolution and final completion. The sync emits a
stable, per-symbol event sequence so a tail of ``liq-data.log`` (or
a CLI stderr heartbeat) shows continuous progress.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    DatabentoBarRecord,
    DatabentoProvider,
    DatabentoTransientError,
)
from liq.data.service import DataService
from liq.data.sync_events import (
    EVENT_SYMBOL_COMPLETED,
    EVENT_SYMBOL_FAILED,
    EVENT_SYMBOL_STARTED,
    EVENT_SYNC_COMPLETED,
    EVENT_SYNC_STARTED,
)
from liq.data.universes import UniverseDefinition, UniverseKind


def _record(symbol: str, day: int) -> DatabentoBarRecord:
    ts = datetime(2024, 6, day, 14, 30, tzinfo=UTC)
    return DatabentoBarRecord(
        ts_event_ns=int(ts.timestamp() * 1e9),
        instrument_id=15144,
        open_q9=int(Decimal("100.00") * DATABENTO_PRICE_SCALE),
        high_q9=int(Decimal("100.10") * DATABENTO_PRICE_SCALE),
        low_q9=int(Decimal("99.90") * DATABENTO_PRICE_SCALE),
        close_q9=int(Decimal("100.05") * DATABENTO_PRICE_SCALE),
        volume=1_000,
        symbol=symbol,
    )


class _FakeStore:
    def __init__(self, records: list[DatabentoBarRecord]) -> None:
        self._records = records
        self.symbology = {
            "mappings": {
                r.symbol: [
                    {
                        "start_date": "2024-01-01",
                        "end_date": "2099-12-31",
                        "symbol": str(r.instrument_id),
                    }
                ]
                for r in records
            }
        }

    def __iter__(self):
        return iter(self._records)


class _FakeTS:
    def __init__(self, store_by_sym: dict[str, list[DatabentoBarRecord]]) -> None:
        self._by_sym = store_by_sym

    def get_range(self, **kwargs) -> _FakeStore:
        sym = kwargs["symbols"][0]
        return _FakeStore(self._by_sym.get(sym, []))


class _FakeBatch:
    def submit_job(self, **kwargs) -> dict:  # pragma: no cover
        return {"id": "job", "state": "received"}

    def get_job_details(self, job_id: str) -> dict:  # pragma: no cover
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs) -> _FakeStore:  # pragma: no cover
        return _FakeStore([])


class _FakeClient:
    def __init__(self, by_sym: dict[str, list[DatabentoBarRecord]]) -> None:
        self.timeseries = _FakeTS(by_sym)
        self.batch = _FakeBatch()


def _patch_databento(by_sym: dict[str, list[DatabentoBarRecord]]):
    def _factory(_settings):
        return DatabentoProvider(api_key="x", client=_FakeClient(by_sym))

    return patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _factory})


def _explicit(symbols: list[str], name: str = "watch") -> UniverseDefinition:
    return UniverseDefinition(
        name=name,
        version=1,
        kind=UniverseKind.EXPLICIT,
        spec={"symbols": symbols},
    )


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from liq.data.settings import get_settings, get_store

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_store.cache_clear()
    yield DataService()
    get_settings.cache_clear()
    get_store.cache_clear()


def _events(records: list[logging.LogRecord], name: str) -> list[logging.LogRecord]:
    return [r for r in records if getattr(r, "event", None) == name]


class TestSyncEnvelope:
    def test_sync_started_and_completed_fire_once_each(
        self, service: DataService, caplog: pytest.LogCaptureFixture
    ) -> None:
        u = _explicit(["AAPL", "MSFT"])
        records = {"AAPL": [_record("AAPL", 3)], "MSFT": [_record("MSFT", 3)]}
        caplog.set_level(logging.INFO, logger="liq.data.service")
        with _patch_databento(records):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert len(_events(caplog.records, EVENT_SYNC_STARTED)) == 1
        assert len(_events(caplog.records, EVENT_SYNC_COMPLETED)) == 1

    def test_sync_envelope_carries_run_id(
        self, service: DataService, caplog: pytest.LogCaptureFixture
    ) -> None:
        u = _explicit(["AAPL"])
        caplog.set_level(logging.INFO, logger="liq.data.service")
        with _patch_databento({"AAPL": [_record("AAPL", 3)]}):
            report = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        started = _events(caplog.records, EVENT_SYNC_STARTED)[0]
        completed = _events(caplog.records, EVENT_SYNC_COMPLETED)[0]
        assert started.sync_run_id == report["sync_run_id"]
        assert completed.sync_run_id == report["sync_run_id"]
        assert completed.symbols == 1
        assert completed.api_calls == 1


class TestSymbolProgress:
    def test_symbol_started_and_completed_per_symbol_with_work(
        self, service: DataService, caplog: pytest.LogCaptureFixture
    ) -> None:
        u = _explicit(["AAPL", "MSFT"])
        records = {"AAPL": [_record("AAPL", 3)], "MSFT": [_record("MSFT", 3)]}
        caplog.set_level(logging.INFO, logger="liq.data.service")
        with _patch_databento(records):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        starts = _events(caplog.records, EVENT_SYMBOL_STARTED)
        completes = _events(caplog.records, EVENT_SYMBOL_COMPLETED)
        assert {r.symbol for r in starts} == {"AAPL", "MSFT"}
        assert {r.symbol for r in completes} == {"AAPL", "MSFT"}

    def test_no_symbol_started_when_already_covered(
        self, service: DataService, caplog: pytest.LogCaptureFixture
    ) -> None:
        u = _explicit(["AAPL"])
        records = {"AAPL": [_record("AAPL", 3)]}
        with _patch_databento(records):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        caplog.clear()
        caplog.set_level(logging.INFO, logger="liq.data.service")
        with _patch_databento(records):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert _events(caplog.records, EVENT_SYMBOL_STARTED) == []
        assert _events(caplog.records, EVENT_SYMBOL_COMPLETED) == []

    def test_symbol_failed_emitted_when_provider_raises(
        self, service: DataService, caplog: pytest.LogCaptureFixture
    ) -> None:
        u = _explicit(["BOOM"])

        def _explodes(_settings):
            class _TS:
                def get_range(self, **kwargs):
                    raise DatabentoTransientError("kaboom")

            class _C:
                timeseries = _TS()
                batch = _FakeBatch()

            return DatabentoProvider(
                api_key="x", client=_C(), sleep_fn=lambda _s: None, max_retry_attempts=1
            )

        caplog.set_level(logging.ERROR, logger="liq.data.service")
        with (
            patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _explodes}),
            pytest.raises(DatabentoTransientError),
        ):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        failed = _events(caplog.records, EVENT_SYMBOL_FAILED)
        assert len(failed) == 1
        assert failed[0].symbol == "BOOM"

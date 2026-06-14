"""Structured batch-progress events from the Databento provider.

Operators running long-span batch fetches need to see the job's
movement through submit → poll → download. The provider emits one
event per state transition; the wrapping ``DataService.sync`` event
catalog lives in :mod:`liq.data.sync_events`.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    DatabentoBarRecord,
    DatabentoProvider,
)
from liq.data.sync_events import (
    EVENT_BATCH_DOWNLOAD_STARTED,
    EVENT_BATCH_POLLING,
    EVENT_BATCH_RESUMED,
    EVENT_BATCH_SUBMITTED,
)


def _record(symbol: str = "AAPL") -> DatabentoBarRecord:
    ts = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    return DatabentoBarRecord(
        ts_event_ns=int(ts.timestamp() * 1e9),
        instrument_id=12345,
        open_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        high_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        low_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        close_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        volume=1,
        symbol=symbol,
    )


class _FakeStore:
    def __init__(self, records: list[DatabentoBarRecord]) -> None:
        self._records = records
        self.symbology = {
            "mappings": {
                r.symbol: [
                    {
                        "start_date": "2025-01-01",
                        "end_date": "2099-12-31",
                        "symbol": str(r.instrument_id),
                    }
                ]
                for r in records
            }
        }

        class _Meta:
            schema = "ohlcv-1m"

        self.metadata = _Meta()

    def __iter__(self):
        return iter(self._records)


class _Batch:
    def __init__(self, store: _FakeStore, *, poll_states: list[str] | None = None) -> None:
        self._store = store
        self._poll_states = list(poll_states or [])
        self.submit_calls: list[dict] = []
        self.download_calls: list[dict] = []

    def submit_job(self, **kwargs) -> dict:
        self.submit_calls.append(kwargs)
        return {"id": "job-7", "state": "received"}

    def get_job_details(self, job_id: str) -> dict:
        if self._poll_states:
            return {"id": job_id, "state": self._poll_states.pop(0)}
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs) -> _FakeStore:
        self.download_calls.append(kwargs)
        return self._store


class _TS:
    def get_range(self, **kwargs):  # pragma: no cover
        raise AssertionError("get_range must not run for batch tests")


class _Client:
    def __init__(self, batch: _Batch) -> None:
        self.timeseries = _TS()
        self.batch = batch


@pytest.fixture
def provider(tmp_path: Path) -> tuple[DatabentoProvider, _Batch]:
    store = _FakeStore([_record()])
    batch = _Batch(store)
    p = DatabentoProvider(
        api_key="x",
        client=_Client(batch),
        batch_threshold_days=1,
        batch_jobs_dir=str(tmp_path / "jobs"),
        sleep_fn=lambda _s: None,
    )
    return p, batch


def _events(records: list[logging.LogRecord], name: str) -> list[logging.LogRecord]:
    return [r for r in records if getattr(r, "event", None) == name]


class TestBatchEvents:
    def test_batch_submitted_fires_on_new_job(
        self,
        provider: tuple[DatabentoProvider, _Batch],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        prov, _batch = provider
        caplog.set_level(logging.INFO, logger="liq.data.providers.databento")
        prov.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        submitted = _events(caplog.records, EVENT_BATCH_SUBMITTED)
        assert len(submitted) == 1
        assert submitted[0].symbol == "AAPL"
        assert submitted[0].job_id == "job-7"

    def test_class_share_symbol_uses_databento_raw_symbol_but_preserves_output_symbol(
        self,
        tmp_path: Path,
    ) -> None:
        store = _FakeStore([_record("BF.B")])
        batch = _Batch(store)
        prov = DatabentoProvider(
            api_key="x",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )

        df = prov.fetch_bars("BF-B", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        assert batch.submit_calls[0]["symbols"] == ["BF.B"]
        assert df["symbol"].to_list() == ["BF-B"]

    def test_batch_polling_fires_per_poll_tick(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = _FakeStore([_record()])
        batch = _Batch(store, poll_states=["received", "queued", "processing", "done"])
        prov = DatabentoProvider(
            api_key="x",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )
        caplog.set_level(logging.INFO, logger="liq.data.providers.databento")
        prov.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        polls = _events(caplog.records, EVENT_BATCH_POLLING)
        states = [r.state for r in polls]
        assert "received" in states
        assert "queued" in states

    def test_batch_download_started_fires_before_download(
        self,
        provider: tuple[DatabentoProvider, _Batch],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        prov, _batch = provider
        caplog.set_level(logging.INFO, logger="liq.data.providers.databento")
        prov.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        starts = _events(caplog.records, EVENT_BATCH_DOWNLOAD_STARTED)
        assert len(starts) == 1
        assert starts[0].job_id == "job-7"

    def test_batch_resumed_fires_on_existing_marker(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = _FakeStore([_record()])
        batch = _Batch(store)
        prov = DatabentoProvider(
            api_key="x",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )

        # First call: marker is written, download succeeds. Force a fresh
        # marker for the second call to exercise resume by injecting a
        # download failure that preserves it.
        class _Failing(_Batch):
            def download(self, **kwargs):  # type: ignore[override]
                raise RuntimeError("boom")

        flaky_batch = _Failing(store)
        prov_flaky = DatabentoProvider(
            api_key="x",
            client=_Client(flaky_batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )
        with pytest.raises(RuntimeError):
            prov_flaky.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert flaky_batch.submit_calls  # initial submission happened

        # Second call: marker exists; we expect ``batch_resumed`` to fire
        # and no new ``submit_job`` to be issued.
        caplog.set_level(logging.INFO, logger="liq.data.providers.databento")
        prov.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        resumed = _events(caplog.records, EVENT_BATCH_RESUMED)
        assert len(resumed) == 1
        # No ``submit_job`` on the second call: the resume path is in
        # the new provider instance, whose ``batch.submit_calls`` is
        # still empty.
        assert batch.submit_calls == []

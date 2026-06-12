"""TDD tests for hardened ``DataService.sync(...)`` behaviour.

Covers:

* Resumability — a per-symbol provider failure rolls back **that
  symbol's** manifest claim but leaves prior-symbol claims persisted,
  so re-running picks up where it left off with no holes or duplicates.
* Missing-bar ambiguity — a gap that the provider fulfils with zero
  bars still produces a manifest row, so subsequent syncs do NOT
  re-bill the venue (FR-4).
* PIT enforcement — ``ResolvedUniverse.pit is False`` emits a
  ``pit_warning`` log and proceeds (sweep mode will reject this
  separately).
* Structured log events — ``universe_resolved``,
  ``manifest_gap_detected``, ``manifest_range_appended``,
  ``manifest_rollback``, ``pit_warning`` all share a single
  ``sync_run_id`` correlation key.
* Concurrency guard — a held sync lock against the same (provider,
  dataset, timeframe, universe) makes a parallel sync raise
  ``SyncLockedError`` instead of corrupting the manifest.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from filelock import FileLock

from liq.data.manifest import CoverageManifest
from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    DatabentoBarRecord,
    DatabentoProvider,
    DatabentoTransientError,
)
from liq.data.service import DataService
from liq.data.sync_events import (
    EVENT_MANIFEST_GAP_DETECTED,
    EVENT_MANIFEST_RANGE_APPENDED,
    EVENT_MANIFEST_ROLLBACK,
    EVENT_PIT_WARNING,
    EVENT_UNIVERSE_RESOLVED,
    SyncLockedError,
)
from liq.data.universes import (
    UniverseDefinition,
    UniverseKind,
)

# ----- fakes ----------------------------------------------------------------


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


class _FakeTimeseries:
    def __init__(self, store_factory) -> None:
        self._store_factory = store_factory
        self.calls: list[dict] = []

    def get_range(self, **kwargs) -> _FakeStore:
        self.calls.append(kwargs)
        return self._store_factory(kwargs)


class _FakeBatch:
    def submit_job(self, **kwargs) -> dict:  # pragma: no cover
        return {"id": "job-1"}

    def get_job_details(self, job_id: str) -> dict:  # pragma: no cover
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs):  # pragma: no cover
        return _FakeStore([])


class _FakeClient:
    def __init__(self, store_factory) -> None:
        self.timeseries = _FakeTimeseries(store_factory)
        self.batch = _FakeBatch()


def _records_by_sym(mapping: dict[str, list[DatabentoBarRecord]]):
    def _factory(call_kwargs: dict) -> _FakeStore:
        sym = call_kwargs["symbols"][0]
        return _FakeStore(mapping.get(sym, []))

    return _factory


def _patch_databento(records: dict[str, list[DatabentoBarRecord]]):
    def _factory(_settings):
        return DatabentoProvider(
            api_key="cassette",
            client=_FakeClient(_records_by_sym(records)),
        )

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


# ----- resumability ---------------------------------------------------------


class TestResumability:
    """A failed sym leaves earlier syms covered → second run completes the rest."""

    def test_partial_failure_persists_committed_symbols(
        self, service: DataService, tmp_path: Path
    ) -> None:
        # First sync: AAPL OK, MSFT explodes after 1 successful symbol.
        good = {"AAPL": [_record("AAPL", 3)]}

        def _flaky_factory(_settings):
            class _TS:
                calls = 0

                def get_range(self, **kwargs):
                    self.calls += 1
                    sym = kwargs["symbols"][0]
                    if sym == "MSFT":
                        raise DatabentoTransientError("boom")
                    return _FakeStore(good[sym])

            class _C:
                timeseries = _TS()
                batch = _FakeBatch()

            return DatabentoProvider(
                api_key="x", client=_C(), sleep_fn=lambda _s: None, max_retry_attempts=1
            )

        u = _explicit(["AAPL", "MSFT"])
        with (
            patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _flaky_factory}),
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

        # AAPL's manifest claim survived; MSFT's rolled back.
        aapl = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        msft = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="MSFT",
        )
        assert len(aapl.ranges) == 1
        assert msft.ranges == []

        # Second sync with the venue healthy — only MSFT hits the wire.
        full = {"AAPL": [_record("AAPL", 3)], "MSFT": [_record("MSFT", 3)]}
        with _patch_databento(full):
            report = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert report["api_calls"] == 1
        assert report["symbols"] == 2

        # A no-op third run proves the resume produced full manifest
        # coverage, and the already-committed first symbol was not
        # duplicated while recovering the failed symbol.
        with _patch_databento(full):
            third = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert third["api_calls"] == 0
        assert third["manifest_gaps"] == 0
        aapl_df = service.load("databento", "AAPL", "1m")
        msft_df = service.load("databento", "MSFT", "1m")
        assert aapl_df.height == 1
        assert msft_df.height == 1
        assert aapl_df.n_unique(subset=["timestamp"]) == 1
        assert msft_df.n_unique(subset=["timestamp"]) == 1


# ----- missing-bar ambiguity ------------------------------------------------


class TestMissingBarAmbiguity:
    """Zero rows ≠ no claim. A fetched gap commits a manifest row even
    when the provider returned no bars (thin/halted symbol)."""

    def test_zero_bars_still_records_manifest_row(
        self, service: DataService, tmp_path: Path
    ) -> None:
        u = _explicit(["THIN"])
        with _patch_databento({"THIN": []}):
            first = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert first["api_calls"] == 1
        assert first["rows_fetched"] == 0

        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="THIN",
        )
        assert len(m.ranges) == 1  # the empty-rows gap is still claimed

        # Second sync — manifest covers it, no fetch.
        with _patch_databento({"THIN": []}):
            second = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert second["api_calls"] == 0
        assert second["manifest_gaps"] == 0


# ----- PIT enforcement ------------------------------------------------------


class TestPITEnforcement:
    def test_non_pit_universe_emits_warning(
        self,
        service: DataService,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        universe = UniverseDefinition(
            name="sp500",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "stub", "id": "SP500"},
        )
        caplog.set_level(logging.WARNING, logger="liq.data.service")
        with _patch_databento({}):
            service.sync(
                universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        pit_records = [r for r in caplog.records if getattr(r, "event", None) == EVENT_PIT_WARNING]
        assert len(pit_records) == 1
        assert getattr(pit_records[0], "universe", None) == "sp500"

    def test_pit_universe_emits_no_warning(
        self,
        service: DataService,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        u = _explicit(["AAPL"])
        caplog.set_level(logging.WARNING, logger="liq.data.service")
        with _patch_databento({"AAPL": [_record("AAPL", 3)]}):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert not [r for r in caplog.records if getattr(r, "event", None) == EVENT_PIT_WARNING]


# ----- structured logging ---------------------------------------------------


class TestStructuredLogging:
    """Every emitted event shares a single ``sync_run_id`` so the operator
    can reconstruct the run."""

    def test_events_share_one_sync_run_id(
        self,
        service: DataService,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        u = _explicit(["AAPL"])
        caplog.set_level(logging.INFO, logger="liq.data.service")
        with _patch_databento({"AAPL": [_record("AAPL", 3)]}):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )

        events = [r for r in caplog.records if getattr(r, "event", None) is not None]
        run_ids = {getattr(r, "sync_run_id", None) for r in events}
        run_ids.discard(None)
        assert len(run_ids) == 1, f"expected one sync_run_id; got {run_ids}"

        event_names = [getattr(r, "event", None) for r in events]
        assert EVENT_UNIVERSE_RESOLVED in event_names
        assert EVENT_MANIFEST_GAP_DETECTED in event_names
        assert EVENT_MANIFEST_RANGE_APPENDED in event_names

    def test_rollback_event_emitted_on_provider_failure(
        self,
        service: DataService,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
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
        rb = [r for r in caplog.records if getattr(r, "event", None) == EVENT_MANIFEST_ROLLBACK]
        assert len(rb) == 1
        assert getattr(rb[0], "symbol", None) == "BOOM"
        assert getattr(rb[0], "error_type", None) == "DatabentoTransientError"


# ----- concurrency guard ----------------------------------------------------


class TestConcurrencyGuard:
    """A held sync lock keyed on (provider, dataset, timeframe, universe)
    causes a parallel sync to raise ``SyncLockedError`` rather than
    racing the manifest."""

    def test_held_lock_blocks_parallel_sync(self, service: DataService, tmp_path: Path) -> None:
        u = _explicit(["AAPL"])
        lock_path = tmp_path / "locks" / "sync" / "databento--EQUS.MINI--1m--watch.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        held = FileLock(str(lock_path))
        held.acquire()
        try:
            with (
                _patch_databento({"AAPL": [_record("AAPL", 3)]}),
                pytest.raises(SyncLockedError),
            ):
                service.sync(
                    u,
                    start=date(2024, 6, 3),
                    end=date(2024, 6, 4),
                    provider="databento",
                    timeframe="1m",
                    dataset="EQUS.MINI",
                    lock_timeout=0.1,
                )
        finally:
            held.release()

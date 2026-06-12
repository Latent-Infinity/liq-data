"""TDD tests for ``DataService.sync(universe, ...)``.

The capability under test: resolve a universe definition, compute the
per-symbol fetch plan from the coverage manifest, dispatch the
provider for each gap, persist symbology, record what landed —
transactionally per symbol so a mid-flight error rolls back the
manifest claim. ``force_refresh=True`` overrides manifest and re-fetches.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from liq.data.manifest import CoverageManifest
from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    DatabentoBarRecord,
    DatabentoTransientError,
)
from liq.data.service import DataService
from liq.data.universes import (
    UniverseDefinition,
    UniverseKind,
    UniverseRegistry,
)

# ----- fixtures --------------------------------------------------------------


def _record(symbol: str, day: int, *, minute: int = 0) -> DatabentoBarRecord:
    ts = datetime(2024, 6, day, 14, 30 + minute, tzinfo=UTC)
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
    """Mimic the bits of databento.DBNStore the provider consumes."""

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
    submit_calls: list[dict]

    def __init__(self) -> None:
        self.submit_calls = []

    def submit_job(self, **kwargs) -> dict:  # pragma: no cover — sync stays below threshold
        self.submit_calls.append(kwargs)
        return {"id": "job-1"}

    def get_job_details(self, job_id: str) -> dict:  # pragma: no cover
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs) -> Any:  # pragma: no cover
        return _FakeStore([])


class _FakeClient:
    def __init__(self, store_factory) -> None:
        self.timeseries = _FakeTimeseries(store_factory)
        self.batch = _FakeBatch()


def _store_factory_per_symbol(records_by_symbol: dict[str, list[DatabentoBarRecord]]):
    def _factory(call_kwargs: dict) -> _FakeStore:
        sym = call_kwargs["symbols"][0]
        return _FakeStore(records_by_symbol.get(sym, []))

    return _factory


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataService:
    # ``get_settings`` / ``get_store`` are ``functools.cache``'d so the
    # first test's ``DATA_ROOT`` env var would otherwise be seen by every
    # subsequent test. Clear before + after so the per-test ``tmp_path``
    # is always honored.
    from liq.data.settings import get_settings, get_store

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_store.cache_clear()
    yield DataService()
    get_settings.cache_clear()
    get_store.cache_clear()


def _explicit_universe(symbols: list[str], name: str = "watch") -> UniverseDefinition:
    return UniverseDefinition(
        name=name,
        version=1,
        kind=UniverseKind.EXPLICIT,
        spec={"symbols": symbols},
    )


def _patch_databento_factory(records_by_symbol: dict[str, list[DatabentoBarRecord]]):
    """Inject a fake databento.Historical via ``DataService._PROVIDER_FACTORIES``."""
    from liq.data.providers.databento import DatabentoProvider

    def _factory(settings):  # noqa: ARG001
        return DatabentoProvider(
            api_key="cassette",
            client=_FakeClient(_store_factory_per_symbol(records_by_symbol)),
        )

    return patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _factory})


# ----- tests ---------------------------------------------------------------


class TestSyncUniverseBasic:
    def test_returns_report_with_api_calls_and_gaps(
        self, service: DataService, tmp_path: Path
    ) -> None:
        u = _explicit_universe(["AAPL", "MSFT"])
        records = {
            "AAPL": [_record("AAPL", 3)],
            "MSFT": [_record("MSFT", 3)],
        }
        with _patch_databento_factory(records):
            report = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert report["api_calls"] == 2  # one per symbol
        assert report["symbols"] == 2
        assert report["rows_fetched"] == 2

    def test_second_sync_makes_zero_api_calls(self, service: DataService, tmp_path: Path) -> None:
        u = _explicit_universe(["AAPL"])
        records = {"AAPL": [_record("AAPL", 3)]}
        with _patch_databento_factory(records):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
            report2 = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        assert report2["api_calls"] == 0
        assert report2["manifest_gaps"] == 0

    def test_growth_only_fetches_new_symbol(self, service: DataService, tmp_path: Path) -> None:
        seed = _explicit_universe(["AAPL"], name="watch")
        records = {
            "AAPL": [_record("AAPL", 3)],
            "MSFT": [_record("MSFT", 3)],
        }
        with _patch_databento_factory(records):
            service.sync(
                seed,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
            grown = _explicit_universe(["AAPL", "MSFT"], name="watch")
            report = service.sync(
                grown,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        # AAPL is fully covered; only MSFT actually hits the venue.
        assert report["api_calls"] == 1
        assert report["symbols"] == 2

    def test_manifest_records_half_open_next_midnight_window(
        self, service: DataService, tmp_path: Path
    ) -> None:
        u = _explicit_universe(["AAPL"])
        records = {"AAPL": [_record("AAPL", 3)]}
        with _patch_databento_factory(records):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )

        manifest = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )

        assert manifest.ranges[0].start == datetime(2024, 6, 3, tzinfo=UTC)
        assert manifest.ranges[0].end == datetime(2024, 6, 5, tzinfo=UTC)

    def test_force_refresh_re_fetches_everything(
        self, service: DataService, tmp_path: Path
    ) -> None:
        u = _explicit_universe(["AAPL"])
        records = {"AAPL": [_record("AAPL", 3)]}
        with _patch_databento_factory(records):
            service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
            forced = service.sync(
                u,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                force_refresh=True,
            )
        assert forced["api_calls"] == 1


class TestSyncRollback:
    def test_provider_failure_rolls_back_manifest(
        self, service: DataService, tmp_path: Path
    ) -> None:
        u = _explicit_universe(["AAPL"])

        def _exploding_factory(settings):  # noqa: ARG001
            from liq.data.providers.databento import (
                DatabentoProvider,
                DatabentoTransientError,
            )

            class _ExplodingTS:
                calls: list[dict] = []

                def get_range(self, **kwargs):
                    self.calls.append(kwargs)
                    raise DatabentoTransientError("boom")

            class _Client:
                timeseries = _ExplodingTS()
                batch = _FakeBatch()

            return DatabentoProvider(
                api_key="x",
                client=_Client(),
                sleep_fn=lambda _s: None,
                max_retry_attempts=1,
            )

        with (
            patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _exploding_factory}),
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
        # Manifest must NOT claim coverage for the failed symbol.
        m = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAPL",
        )
        assert m.ranges == []


class TestSyncRegistryShortcut:
    def test_sync_accepts_universe_name_via_registry(
        self, service: DataService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = UniverseRegistry(tmp_path)
        reg.save(_explicit_universe(["AAPL"]))
        records = {"AAPL": [_record("AAPL", 3)]}
        with _patch_databento_factory(records):
            report = service.sync(
                "watch",
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                registry=reg,
            )
        assert report["api_calls"] == 1

    def test_sync_resolves_set_op_inputs_from_registry(
        self, service: DataService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = UniverseRegistry(tmp_path)
        reg.save(_explicit_universe(["AAPL"], name="watch"))
        reg.save(_explicit_universe(["MSFT"], name="extra"))
        reg.save(
            UniverseDefinition(
                name="combo",
                version=1,
                kind=UniverseKind.SET_OP,
                spec={"op": "union", "inputs": ["watch", "extra"]},
            )
        )
        records = {
            "AAPL": [_record("AAPL", 3)],
            "MSFT": [_record("MSFT", 3)],
        }

        with _patch_databento_factory(records):
            report = service.sync(
                "combo",
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                registry=reg,
            )

        assert report["symbols"] == 2
        assert report["api_calls"] == 2

    def test_sync_composite_defaults_to_non_pit_stub(
        self, service: DataService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        universe = UniverseDefinition(
            name="sp500",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "stub", "id": "SP500"},
        )

        with _patch_databento_factory({}):
            report = service.sync(
                universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )

        assert report["symbols"] == 0
        assert report["api_calls"] == 0
        assert report["pit"] is False

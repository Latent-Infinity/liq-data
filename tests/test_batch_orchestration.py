"""Batch-orchestrated universe sync tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from typer.testing import CliRunner

from liq.data.cli import app
from liq.data.exceptions import ProviderNoDataError
from liq.data.protocols import BatchJob
from liq.data.service import DataService
from liq.data.universes import UniverseDefinition, UniverseKind


def _bars(symbol: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2, 14, 30, tzinfo=UTC)],
            "open": [Decimal("100.00")],
            "high": [Decimal("101.00")],
            "low": [Decimal("99.00")],
            "close": [Decimal("100.50")],
            "volume": [Decimal("1000")],
            "symbol": [symbol],
        }
    )


class _BatchProvider:
    name = "fakebatch"
    supported_asset_classes = ["equity"]
    batch_poll_seconds = 0.0

    def __init__(self) -> None:
        self.actions: list[tuple[str, str]] = []
        self.active = 0
        self.max_active = 0

    def fetch_bars(self, symbol: str, start: date, end: date, timeframe: str = "1d"):
        raise AssertionError("serial fetch_bars should not run in batch orchestration")

    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        return pl.DataFrame(schema={"symbol": pl.Utf8})

    def submit_batch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
        *,
        dataset: str,
        sync_run_id: str | None = None,
    ) -> BatchJob:
        self.actions.append(("submit", symbol))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        return BatchJob(
            provider=self.name,
            job_id=f"job-{symbol}",
            signature=f"sig-{symbol}",
            dataset=dataset,
            symbol=symbol,
            timeframe=timeframe,
            start=datetime(start.year, start.month, start.day, tzinfo=UTC),
            end=datetime(end.year, end.month, end.day, tzinfo=UTC),
        )

    def poll_batch_bars(self, job: BatchJob, *, sync_run_id: str | None = None) -> bool:
        self.actions.append(("poll", job.symbol))
        return True

    def fetch_completed_batch_bars(
        self, job: BatchJob, *, sync_run_id: str | None = None
    ) -> pl.DataFrame:
        self.actions.append(("download", job.symbol))
        self.active -= 1
        return _bars(job.symbol)


@pytest.fixture
def service_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from liq.data.settings import get_settings, get_store

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_store.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
    get_store.cache_clear()


def _universe(symbols: list[str]) -> UniverseDefinition:
    return UniverseDefinition(
        name="batch-watch",
        version=1,
        kind=UniverseKind.EXPLICIT,
        spec={"symbols": symbols},
    )


def test_sync_batch_keeps_bounded_provider_jobs_in_flight(service_env: Path) -> None:
    provider = _BatchProvider()

    def _factory(_settings: Any) -> _BatchProvider:
        return provider

    with patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}):
        report = DataService().sync_batch(
            _universe(["A", "B", "C"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="fakebatch",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=2,
            poll_seconds=0,
        )

    assert report["orchestration"] == "batch"
    assert report["api_calls"] == 3
    assert report["rows_fetched"] == 3
    assert provider.max_active == 2
    assert provider.actions[:2] == [("submit", "A"), ("submit", "B")]
    assert ("submit", "C") in provider.actions


def test_cli_batch_orchestration_routes_to_batch_sync(service_env: Path) -> None:
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "universe",
            "create",
            "--name",
            "batch-watch",
            "--kind",
            "explicit",
            "--symbols",
            "A,B",
        ],
    )
    provider = _BatchProvider()

    def _factory(_settings: Any) -> _BatchProvider:
        return provider

    with patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}):
        result = runner.invoke(
            app,
            [
                "sync",
                "batch-watch",
                "--start",
                "2024-01-02",
                "--end",
                "2024-01-03",
                "--provider",
                "fakebatch",
                "--dataset",
                "FAKE.DATA",
                "--orchestration",
                "batch",
                "--max-in-flight",
                "2",
            ],
        )

    assert result.exit_code == 0, result.output
    assert '"orchestration": "batch"' in result.output
    assert provider.actions[:2] == [("submit", "A"), ("submit", "B")]


# ----- failure / edge-case tests -------------------------------------------


def test_sync_batch_rejects_non_batch_provider(service_env: Path) -> None:
    """A provider missing the BatchMarketDataProvider triple must be
    rejected up front rather than silently failing at submit time."""

    class _SerialOnlyProvider:
        name = "serialonly"
        supported_asset_classes = ["equity"]

        def fetch_bars(self, *_a: Any, **_k: Any) -> pl.DataFrame:  # pragma: no cover
            return pl.DataFrame()

    def _factory(_settings: Any) -> _SerialOnlyProvider:
        return _SerialOnlyProvider()

    with (
        patch.dict(DataService._PROVIDER_FACTORIES, {"serialonly": _factory}),
        pytest.raises(ValueError, match="batch orchestration"),
    ):
        DataService().sync_batch(
            _universe(["A"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="serialonly",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=2,
            poll_seconds=0,
        )


def test_sync_batch_rejects_max_in_flight_below_one(service_env: Path) -> None:
    provider = _BatchProvider()

    def _factory(_settings: Any) -> _BatchProvider:
        return provider

    with (
        patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}),
        pytest.raises(ValueError, match="max_in_flight"),
    ):
        DataService().sync_batch(
            _universe(["A"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="fakebatch",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=0,
            poll_seconds=0,
        )


def test_sync_batch_max_in_flight_one_serializes_jobs(service_env: Path) -> None:
    """A degenerate ``max_in_flight=1`` must still complete and never
    submit two jobs concurrently."""
    provider = _BatchProvider()

    def _factory(_settings: Any) -> _BatchProvider:
        return provider

    with patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}):
        report = DataService().sync_batch(
            _universe(["A", "B", "C"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="fakebatch",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=1,
            poll_seconds=0,
        )

    assert report["rows_fetched"] == 3
    assert provider.max_active == 1


def test_sync_batch_download_failure_emits_symbol_failed_and_rolls_back(
    service_env: Path,
    caplog,
) -> None:
    """A failure in ``fetch_completed_batch_bars`` must surface both
    ``manifest_rollback`` and ``symbol_failed`` events with the right
    symbol, then re-raise so the caller can decide policy."""
    import logging

    class _FailingDownload(_BatchProvider):
        def fetch_completed_batch_bars(self, job, *, sync_run_id=None):  # type: ignore[override]
            raise RuntimeError("download blew up")

    provider = _FailingDownload()

    def _factory(_settings: Any) -> _FailingDownload:
        return provider

    caplog.set_level(logging.ERROR, logger="liq.data.service")
    with (
        patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}),
        pytest.raises(RuntimeError, match="download blew up"),
    ):
        DataService().sync_batch(
            _universe(["A"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="fakebatch",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=1,
            poll_seconds=0,
        )

    events = [getattr(r, "event", None) for r in caplog.records]
    assert "manifest_rollback" in events
    assert "symbol_failed" in events


def test_sync_batch_submit_failure_emits_symbol_failed(service_env: Path, caplog) -> None:
    """A failure on ``submit_batch_bars`` must produce a ``symbol_failed``
    event before raising — otherwise operators can't tell which symbol
    aborted the orchestration."""
    import logging

    class _FailingSubmit(_BatchProvider):
        def submit_batch_bars(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("submit refused")

    provider = _FailingSubmit()

    def _factory(_settings: Any) -> _FailingSubmit:
        return provider

    caplog.set_level(logging.ERROR, logger="liq.data.service")
    with (
        patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}),
        pytest.raises(RuntimeError, match="submit refused"),
    ):
        DataService().sync_batch(
            _universe(["A"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="fakebatch",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=1,
            poll_seconds=0,
        )

    failed = [r for r in caplog.records if getattr(r, "event", None) == "symbol_failed"]
    assert failed and getattr(failed[0], "symbol", None) == "A"


def test_sync_batch_provider_no_data_submit_skips_symbol(
    service_env: Path,
    caplog,
) -> None:
    """Provider-confirmed no-data symbols should not abort a long universe
    batch run. They are counted and reported so operators can distinguish
    real no-coverage from successfully fetched data."""
    import logging

    class _NoDataThenOk(_BatchProvider):
        def submit_batch_bars(self, symbol, *args, **kwargs):  # type: ignore[override]
            if symbol == "MISSING":
                raise ProviderNoDataError("No data was found for the request")
            return super().submit_batch_bars(symbol, *args, **kwargs)

    provider = _NoDataThenOk()

    def _factory(_settings: Any) -> _NoDataThenOk:
        return provider

    caplog.set_level(logging.INFO, logger="liq.data.service")
    with patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}):
        report = DataService().sync_batch(
            _universe(["MISSING", "OK"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="fakebatch",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=1,
            poll_seconds=0,
        )

    assert report["api_calls"] == 1
    assert report["symbols_skipped"] == 1
    assert report["skipped_symbols"] == ["MISSING"]
    assert ("download", "OK") in provider.actions
    failed = [r for r in caplog.records if getattr(r, "event", None) == "symbol_failed"]
    assert failed and getattr(failed[0], "symbol", None) == "MISSING"


def test_sync_batch_polling_does_not_consume_rate_limit(service_env: Path, monkeypatch) -> None:
    """Polling is a status check, not a billable data request. It must
    not burn rate-limit budget — otherwise a high ``max_in_flight``
    immediately throttles itself."""
    from liq.data import service as svc_mod

    acquire_calls = []

    class _CountingLimiter:
        def __init__(self, *_a, **_k) -> None:
            pass

        def acquire(self) -> None:
            acquire_calls.append(True)

    monkeypatch.setattr(svc_mod, "RateLimiter", _CountingLimiter)

    # Make polling never ready on the first round so the loop polls
    # multiple times before any download fires.
    class _SlowPolling(_BatchProvider):
        def __init__(self) -> None:
            super().__init__()
            self._poll_calls = 0

        def poll_batch_bars(self, job, *, sync_run_id=None):  # type: ignore[override]
            self._poll_calls += 1
            self.actions.append(("poll", job.symbol))
            # Ready only on the 3rd poll for each job.
            return self._poll_calls > len(["A", "B"]) * 2

    provider = _SlowPolling()

    def _factory(_settings: Any) -> _SlowPolling:
        return provider

    with patch.dict(DataService._PROVIDER_FACTORIES, {"fakebatch": _factory}):
        DataService().sync_batch(
            _universe(["A", "B"]),
            start=date(2024, 1, 2),
            end=date(2024, 1, 3),
            provider="fakebatch",
            timeframe="1m",
            dataset="FAKE.DATA",
            max_in_flight=2,
            poll_seconds=0,
        )

    # 2 submits + 2 downloads should burn 4 acquire() calls. Polling
    # must not contribute — even though we polled several times per job.
    assert len(acquire_calls) == 4, (
        f"expected 4 rate-limited acquire() calls (2 submit + 2 download); got {len(acquire_calls)}"
    )

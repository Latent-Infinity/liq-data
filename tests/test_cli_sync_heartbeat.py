"""Smoke test for the ``liq-data sync --verbose`` heartbeat.

The flag installs a stderr handler that prints one short line per
structured sync event. We confirm that an end-to-end CLI invocation
produces the expected envelope events on stderr without breaking the
JSON-on-stdout contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from liq.data.cli import app
from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    DatabentoBarRecord,
    DatabentoProvider,
)
from liq.data.service import DataService

_runner = CliRunner()


def _record(symbol: str) -> DatabentoBarRecord:
    ts = datetime(2024, 6, 3, 14, 30, tzinfo=UTC)
    return DatabentoBarRecord(
        ts_event_ns=int(ts.timestamp() * 1e9),
        instrument_id=12345,
        open_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        high_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        low_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        close_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
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


class _TS:
    def __init__(self, by_sym: dict[str, list[DatabentoBarRecord]]) -> None:
        self._by_sym = by_sym

    def get_range(self, **kwargs):
        return _FakeStore(self._by_sym.get(kwargs["symbols"][0], []))


class _Batch:
    def submit_job(self, **kwargs):  # pragma: no cover
        return {"id": "j", "state": "received"}

    def get_job_details(self, job_id):  # pragma: no cover
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs):  # pragma: no cover
        return _FakeStore([])


class _Client:
    def __init__(self, by_sym):
        self.timeseries = _TS(by_sym)
        self.batch = _Batch()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import logging as _logging

    from liq.data.settings import get_settings, get_store

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_store.cache_clear()

    # The verbose flag mutates the global ``liq.data`` logger. Snapshot
    # the level + handlers so teardown restores them; otherwise the
    # heartbeat handler leaks into other tests' caplog records.
    parent = _logging.getLogger("liq.data")
    saved_level = parent.level
    saved_handlers = list(parent.handlers)

    yield tmp_path

    get_settings.cache_clear()
    get_store.cache_clear()
    parent.setLevel(saved_level)
    parent.handlers = saved_handlers


def test_verbose_flag_prints_event_lines_to_stderr(env: Path) -> None:
    # Seed registry + patch databento provider factory.
    _runner.invoke(
        app,
        [
            "universe",
            "create",
            "--name",
            "watch",
            "--kind",
            "explicit",
            "--symbols",
            "AAPL",
        ],
    )

    def _factory(_settings):
        return DatabentoProvider(api_key="x", client=_Client({"AAPL": [_record("AAPL")]}))

    with patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _factory}):
        result = _runner.invoke(
            app,
            [
                "sync",
                "watch",
                "--start",
                "2024-06-03",
                "--end",
                "2024-06-04",
                "--verbose",
            ],
        )

    assert result.exit_code == 0, result.output
    # When mix_stderr is the default, output captures both — assert
    # both the JSON envelope (stdout) and the event lines (stderr).
    out = result.output
    assert "[sync_started]" in out
    assert "[universe_resolved]" in out
    assert "[symbol_started] symbol=AAPL" in out
    assert "[symbol_completed] symbol=AAPL" in out
    assert "[sync_completed]" in out
    # JSON envelope: at least one JSON-shaped report line.
    assert any(line.startswith("{") for line in out.splitlines())

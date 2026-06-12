"""CLI tests for the universe + sync subcommands."""

from __future__ import annotations

import json
from collections.abc import Iterator
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


# ----- inline cassette client (mirrors test_cli_databento) -----------------


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
    def __init__(self, records_by_sym: dict[str, list[DatabentoBarRecord]]) -> None:
        self._records_by_sym = records_by_sym
        self.calls: list[dict] = []

    def get_range(self, **kwargs) -> _FakeStore:
        self.calls.append(kwargs)
        sym = kwargs["symbols"][0]
        return _FakeStore(self._records_by_sym.get(sym, []))


class _FakeBatch:
    def submit_job(self, **kwargs) -> dict:  # pragma: no cover
        return {"id": "job-1"}

    def get_job_details(self, job_id: str) -> dict:  # pragma: no cover
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs) -> _FakeStore:  # pragma: no cover
        return _FakeStore([])


class _FakeClient:
    def __init__(self, records_by_sym: dict[str, list[DatabentoBarRecord]]) -> None:
        self.timeseries = _FakeTimeseries(records_by_sym)
        self.batch = _FakeBatch()


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


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from liq.data.settings import get_settings, get_store

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    get_settings.cache_clear()
    get_store.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
    get_store.cache_clear()


# ----- universe CRUD --------------------------------------------------------


class TestUniverseCRUD:
    def test_create_list_resolve_delete_round_trip(self, env: Path) -> None:
        create = _runner.invoke(
            app,
            [
                "universe",
                "create",
                "--name",
                "watch",
                "--kind",
                "explicit",
                "--symbols",
                "AAPL,MSFT",
            ],
        )
        assert create.exit_code == 0, create.output
        assert "watch" in create.output

        listed = _runner.invoke(app, ["universe", "list"])
        assert listed.exit_code == 0
        assert "watch" in json.loads(listed.output)["names"]

        resolved = _runner.invoke(app, ["universe", "resolve", "watch", "--as-of", "2025-01-02"])
        assert resolved.exit_code == 0
        payload = json.loads(resolved.output)
        assert payload["symbols"] == ["AAPL", "MSFT"]
        assert payload["pit"] is True

        deleted = _runner.invoke(app, ["universe", "delete", "watch"])
        assert deleted.exit_code == 0
        assert json.loads(deleted.output)["removed"] is True

    def test_create_collision_exits_nonzero(self, env: Path) -> None:
        first = _runner.invoke(
            app,
            [
                "universe",
                "create",
                "--name",
                "x",
                "--kind",
                "explicit",
                "--symbols",
                "A",
            ],
        )
        assert first.exit_code == 0
        # Same name, different spec, no overwrite → conflict.
        second = _runner.invoke(
            app,
            [
                "universe",
                "create",
                "--name",
                "x",
                "--kind",
                "explicit",
                "--symbols",
                "B",
                "--version",
                "2",
            ],
        )
        assert second.exit_code == 1

    def test_resolve_missing_exits_nonzero(self, env: Path) -> None:
        result = _runner.invoke(app, ["universe", "resolve", "nope", "--as-of", "2025-01-02"])
        assert result.exit_code == 1

    def test_create_missing_symbols_for_explicit(self, env: Path) -> None:
        result = _runner.invoke(app, ["universe", "create", "--name", "x", "--kind", "explicit"])
        assert result.exit_code == 1
        assert "symbols" in result.output

    def test_create_missing_expr_for_filter(self, env: Path) -> None:
        result = _runner.invoke(app, ["universe", "create", "--name", "x", "--kind", "filter"])
        assert result.exit_code == 1
        assert "expr" in result.output

    def test_create_missing_composite_args(self, env: Path) -> None:
        result = _runner.invoke(app, ["universe", "create", "--name", "x", "--kind", "composite"])
        assert result.exit_code == 1

    def test_create_missing_set_op_args(self, env: Path) -> None:
        result = _runner.invoke(app, ["universe", "create", "--name", "x", "--kind", "set_op"])
        assert result.exit_code == 1

    def test_create_unknown_kind(self, env: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "universe",
                "create",
                "--name",
                "x",
                "--kind",
                "bogus",
                "--symbols",
                "A",
            ],
        )
        assert result.exit_code == 1

    def test_create_filter_round_trips(self, env: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "universe",
                "create",
                "--name",
                "liquid",
                "--kind",
                "filter",
                "--expr",
                "price > 5",
            ],
        )
        assert result.exit_code == 0

    def test_create_composite_round_trips(self, env: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "universe",
                "create",
                "--name",
                "sp500",
                "--kind",
                "composite",
                "--source",
                "stub",
                "--id",
                "SP500",
            ],
        )
        assert result.exit_code == 0

    def test_create_set_op_round_trips(self, env: Path) -> None:
        result = _runner.invoke(
            app,
            [
                "universe",
                "create",
                "--name",
                "combo",
                "--kind",
                "set_op",
                "--op",
                "union",
                "--inputs",
                "a,b",
            ],
        )
        assert result.exit_code == 0


# ----- sync ---------------------------------------------------------------


class TestSyncCli:
    def _patch_factory(self, records: dict[str, list[DatabentoBarRecord]]):
        def _factory(_settings):
            return DatabentoProvider(
                api_key="cassette",
                client=_FakeClient(records),
            )

        return patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _factory})

    def test_sync_returns_json_report(self, env: Path) -> None:
        create = _runner.invoke(
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
        assert create.exit_code == 0

        with self._patch_factory({"AAPL": [_record("AAPL", 3)]}):
            sync = _runner.invoke(
                app,
                [
                    "sync",
                    "watch",
                    "--start",
                    "2024-06-03",
                    "--end",
                    "2024-06-04",
                    "--provider",
                    "databento",
                    "--timeframe",
                    "1m",
                    "--dataset",
                    "EQUS.MINI",
                ],
            )
        assert sync.exit_code == 0, sync.output
        report = json.loads(sync.output)
        assert report["api_calls"] == 1
        assert report["symbols"] == 1

    def test_sync_second_call_is_zero_api_calls(self, env: Path) -> None:
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
        with self._patch_factory({"AAPL": [_record("AAPL", 3)]}):
            _runner.invoke(
                app,
                [
                    "sync",
                    "watch",
                    "--start",
                    "2024-06-03",
                    "--end",
                    "2024-06-04",
                ],
            )
            second = _runner.invoke(
                app,
                [
                    "sync",
                    "watch",
                    "--start",
                    "2024-06-03",
                    "--end",
                    "2024-06-04",
                ],
            )
        report = json.loads(second.output)
        assert report["api_calls"] == 0
        assert report["manifest_gaps"] == 0

    def test_sync_force_refresh_requires_budget_ack(self, env: Path) -> None:
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
        result = _runner.invoke(
            app,
            [
                "sync",
                "watch",
                "--start",
                "2024-06-03",
                "--end",
                "2024-06-04",
                "--force-refresh",
            ],
        )
        assert result.exit_code == 1
        assert "--i-have-budget-authorization" in result.output

    def test_sync_missing_universe_exits_nonzero(self, env: Path) -> None:
        result = _runner.invoke(
            app, ["sync", "nope", "--start", "2024-06-03", "--end", "2024-06-04"]
        )
        assert result.exit_code == 1

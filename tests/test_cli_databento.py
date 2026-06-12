"""CLI smoke tests for the Databento provider path.

These tests cassette-inject a fake ``databento.Historical`` client so
``uv run liq-data fetch --provider databento ...`` returns exit 0
without touching the network. The whole pipeline runs end-to-end:
typer command → ``DataService.fetch`` → ``DataService._get_provider``
→ ``create_databento_provider`` (patched here) → ``DatabentoProvider``
→ in-memory fake DBN store → bar DataFrame → CLI output.

We patch at the factory layer (not at ``DataService.fetch`` itself)
because the criterion is "exit 0 against cassette fake injected" —
patching higher up the stack would skip the provider's own routing /
normalization code paths, defeating the smoke.

The fake DBN store is inlined here (not imported from the provider
test module) so this file stays standalone and doesn't depend on a
``tests`` package being importable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from liq.data.cli import app
from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    SYMBOLOGY_KEY,
    DatabentoBarRecord,
    DatabentoProvider,
)
from liq.store.parquet import ParquetStore

_runner = CliRunner()


# ----- inline cassette fakes ------------------------------------------------


@dataclass
class _FakeDBNStore:
    records: list[DatabentoBarRecord]
    symbology: dict

    def __iter__(self):
        return iter(self.records)


class _FakeTimeseries:
    def __init__(self, store: _FakeDBNStore) -> None:
        self._store = store

    def get_range(self, **_kwargs) -> _FakeDBNStore:
        return self._store


class _FakeBatch:
    def __init__(self, store: _FakeDBNStore) -> None:
        self._store = store

    def submit_job(self, **_kwargs) -> dict:
        return {"id": "job-1"}

    def download(self, **_kwargs) -> _FakeDBNStore:
        return self._store


class _FakeClient:
    def __init__(self, store: _FakeDBNStore) -> None:
        self.timeseries = _FakeTimeseries(store)
        self.batch = _FakeBatch(store)


# ----- canned response builder ----------------------------------------------


def _fake_provider() -> DatabentoProvider:
    """Build a ``DatabentoProvider`` wired to an in-memory fake client.

    Mirrors the production constructor path (api_key + injected client)
    so the CLI exercises the real provider class — only the wire layer
    is fake.
    """
    base_ns = int(datetime(2024, 6, 3, 14, 30, tzinfo=UTC).timestamp() * 1e9)
    records = [
        DatabentoBarRecord(
            ts_event_ns=base_ns + i * 60 * 1_000_000_000,
            instrument_id=15144,
            open_q9=int(Decimal("528.00") * DATABENTO_PRICE_SCALE),
            high_q9=int(Decimal("528.50") * DATABENTO_PRICE_SCALE),
            low_q9=int(Decimal("527.90") * DATABENTO_PRICE_SCALE),
            close_q9=int(Decimal("528.25") * DATABENTO_PRICE_SCALE),
            volume=1_000 + i,
            symbol="SPY",
        )
        for i in range(3)
    ]
    store = _FakeDBNStore(
        records=records,
        symbology={
            "mappings": {
                "SPY": [
                    {
                        "start_date": "2024-06-03",
                        "end_date": "2024-06-05",
                        "symbol": "15144",
                    }
                ],
            },
        },
    )
    return DatabentoProvider(api_key="cassette-fake-key", client=_FakeClient(store))


@pytest.fixture
def factory_calls() -> Iterator[list[object]]:
    """Inject the cassette provider into ``DataService._PROVIDER_FACTORIES``.

    The factory dict captures function references at class-definition
    time, so patching the module-level name in ``liq.data.service`` is
    a no-op. Instead we override the dict entry directly via
    ``patch.dict``, which restores cleanly on test exit.

    Yields a list that receives one entry per factory invocation; tests
    assert on the length to verify the cassette path was taken.
    """
    from liq.data.service import DataService
    from liq.data.settings import get_settings, get_store

    calls: list[object] = []

    def _factory(settings):
        calls.append(settings)
        return _fake_provider()

    get_settings.cache_clear()
    get_store.cache_clear()
    with patch.dict(DataService._PROVIDER_FACTORIES, {"databento": _factory}):
        try:
            yield calls
        finally:
            get_settings.cache_clear()
            get_store.cache_clear()


# ----- tests ----------------------------------------------------------------


class TestFetchCliWithDatabentoProvider:
    def test_fetch_dry_run_returns_zero(self, factory_calls: list[object]) -> None:
        """``--dry-run`` exercises the provider without writing to disk."""
        result = _runner.invoke(
            app,
            [
                "fetch",
                "databento",
                "SPY",
                "--start",
                "2024-06-03",
                "--end",
                "2024-06-04",
                "--timeframe",
                "1m",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Fetched" in result.output
        assert "SPY" in result.output
        assert len(factory_calls) == 1

    def test_fetch_writes_through_to_store(
        self,
        factory_calls: list[object],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without ``--dry-run`` the fetched frame round-trips through liq-store."""
        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        result = _runner.invoke(
            app,
            [
                "fetch",
                "databento",
                "SPY",
                "--start",
                "2024-06-03",
                "--end",
                "2024-06-04",
                "--timeframe",
                "1m",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Stored" in result.output
        assert len(factory_calls) == 1
        store = ParquetStore(str(tmp_path))
        assert store.exists(SYMBOLOGY_KEY)
        symbology = store.read(SYMBOLOGY_KEY)
        assert not symbology.is_empty()
        assert symbology.row(0, named=True)["raw_symbol"] == "SPY"

    def test_fetch_accepts_documented_option_shape(
        self,
        factory_calls: list[object],
    ) -> None:
        result = _runner.invoke(
            app,
            [
                "fetch",
                "--provider",
                "databento",
                "--symbols",
                "SPY",
                "--start",
                "2024-06-03",
                "--end",
                "2024-06-04",
                "--timeframe",
                "1m",
                "--output",
                "json",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert '"provider": "databento"' in result.output
        assert '"symbol": "SPY"' in result.output
        assert '"rows": 3' in result.output
        assert len(factory_calls) == 1

    def test_fetch_renders_databento_in_provider_help(self) -> None:
        """``fetch --help`` should advertise the databento provider name."""
        result = _runner.invoke(app, ["fetch", "--help"])
        assert result.exit_code == 0
        assert "databento" in result.output

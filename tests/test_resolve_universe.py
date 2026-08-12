"""Tests for ``DataService.resolve_universe`` convenience method.

The method is a thin wrapper over ``UniverseResolver.resolve`` that
hides the construction details so the scanner doesn't have to import
the resolver / source / registry trio.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from liq.data.service import DataService
from liq.data.universes import (
    UniverseDefinition,
    UniverseKind,
    UniverseRegistry,
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


class TestResolveUniverse:
    def test_accepts_definition_object(self, service: DataService) -> None:
        defn = UniverseDefinition(
            name="watch",
            version=1,
            kind=UniverseKind.EXPLICIT,
            spec={"symbols": ["AAPL", "MSFT"]},
        )
        resolved = service.resolve_universe(defn, as_of=date(2024, 6, 3))
        assert resolved.symbols == ["AAPL", "MSFT"]
        assert resolved.pit is True
        assert resolved.as_of == date(2024, 6, 3)
        assert resolved.definition_name == "watch"

    def test_accepts_list_of_symbols(self, service: DataService) -> None:
        resolved = service.resolve_universe(["AAPL", "tsla"], as_of=date(2024, 6, 3))
        assert resolved.symbols == ["AAPL", "TSLA"]
        assert resolved.pit is True

    def test_accepts_name_via_registry(self, service: DataService, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        defn = UniverseDefinition(
            name="watch",
            version=1,
            kind=UniverseKind.EXPLICIT,
            spec={"symbols": ["AAPL"]},
        )
        reg.save(defn)
        resolved = service.resolve_universe("watch", as_of=date(2024, 6, 3), registry=reg)
        assert resolved.symbols == ["AAPL"]

    def test_name_without_registry_raises(self, service: DataService) -> None:
        with pytest.raises(ValueError, match="UniverseRegistry"):
            service.resolve_universe("missing", as_of=date(2024, 6, 3))


class TestResolveUniversePointInTime:
    """Composite universes resolve point-in-time when membership snapshots exist."""

    @staticmethod
    def _composite() -> UniverseDefinition:
        return UniverseDefinition(
            name="idx",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "snapshot", "id": "sp500"},
        )

    @staticmethod
    def _write_snapshot(data_root: Path) -> None:
        snap_dir = data_root / "reference" / "universes" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "date": [date(2005, 1, 3), date(2005, 3, 1)],
                "tickers": [["AAA", "BBB", "CCC"], ["AAA", "EEE", "DDD"]],
            }
        ).write_parquet(snap_dir / "sp500.parquet")

    def test_composite_is_point_in_time_when_snapshot_present(
        self, service: DataService, tmp_path: Path
    ) -> None:
        self._write_snapshot(tmp_path)
        resolved = service.resolve_universe(self._composite(), as_of=date(2005, 2, 1))
        assert resolved.pit is True
        assert resolved.symbols == ["AAA", "BBB", "CCC"]  # floor at 2005-01-03

    def test_snapshot_source_without_snapshot_is_fail_closed(self, service: DataService) -> None:
        from liq.data.universes import UniverseResolutionError

        with pytest.raises(UniverseResolutionError, match="no point-in-time membership snapshot"):
            service.resolve_universe(self._composite(), as_of=date(2005, 2, 1))

    def test_explicit_stub_source_remains_current_only(self, service: DataService) -> None:
        current_only = UniverseDefinition(
            name="idx",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "stub", "id": "sp500"},
        )
        resolved = service.resolve_universe(current_only, as_of=date(2005, 2, 1))
        assert resolved.pit is False
        assert resolved.symbols == []

    def test_stub_source_does_not_consume_snapshot_file(
        self, service: DataService, tmp_path: Path
    ) -> None:
        self._write_snapshot(tmp_path)
        current_only = UniverseDefinition(
            name="idx",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "stub", "id": "sp500"},
        )
        resolved = service.resolve_universe(current_only, as_of=date(2005, 2, 1))
        assert resolved.pit is False
        assert resolved.symbols == []

    def test_present_snapshot_dir_missing_id_is_fail_closed(
        self, service: DataService, tmp_path: Path
    ) -> None:
        from liq.data.universes import UniverseResolutionError

        (tmp_path / "reference" / "universes" / "snapshots").mkdir(parents=True)
        with pytest.raises(UniverseResolutionError, match="no point-in-time membership snapshot"):
            service.resolve_universe(self._composite(), as_of=date(2005, 2, 1))

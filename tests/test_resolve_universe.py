"""Tests for ``DataService.resolve_universe`` convenience method.

The method is a thin wrapper over ``UniverseResolver.resolve`` that
hides the construction details so the scanner doesn't have to import
the resolver / source / registry trio.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

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

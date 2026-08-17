"""Tests for ``DataService.resolve_universe`` convenience method.

The method is a thin wrapper over ``UniverseResolver.resolve`` that
hides the construction details so the scanner doesn't have to import
the resolver / source / registry trio.
"""

from __future__ import annotations

import os
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


class TestResolveNamedSnapshotUniverse:
    """A named YAML declaration resolves point-in-time through the snapshot route.

    This pins the full path a caller takes when it only holds a universe
    *name*: ``{data_root}/reference/universes/sp500-pit.yaml`` →
    ``UniverseRegistry`` → ``DataService.resolve_universe`` →
    ``DirectorySnapshotSource`` → ``pit=True`` membership with floor
    semantics — with a missing per-id snapshot parquet fail-closed, never a
    silent current-membership fallback.
    """

    NAME = "sp500-pit"

    @classmethod
    def _definition(cls) -> UniverseDefinition:
        return UniverseDefinition(
            name=cls.NAME,
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "snapshot", "id": "sp500"},
        )

    @classmethod
    def _write_registry_yaml(cls, data_root: Path) -> UniverseRegistry:
        registry = UniverseRegistry(data_root)
        registry.save(cls._definition())
        return registry

    @staticmethod
    def _write_snapshot(data_root: Path) -> None:
        snap_dir = data_root / "reference" / "universes" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "date": [date(2017, 1, 3), date(2017, 6, 1)],
                "tickers": [["AAA", "BBB", "CCC"], ["AAA", "BBB", "DDD"]],
            }
        ).write_parquet(snap_dir / "sp500.parquet")

    def test_named_declaration_resolves_point_in_time(
        self, service: DataService, tmp_path: Path
    ) -> None:
        self._write_snapshot(tmp_path)
        registry = self._write_registry_yaml(tmp_path)
        resolved = service.resolve_universe(self.NAME, as_of=date(2017, 2, 1), registry=registry)
        assert resolved.pit is True
        assert resolved.definition_name == self.NAME
        assert resolved.symbols == ["AAA", "BBB", "CCC"]

    def test_membership_floor_semantics_by_as_of(
        self, service: DataService, tmp_path: Path
    ) -> None:
        self._write_snapshot(tmp_path)
        registry = self._write_registry_yaml(tmp_path)
        # Exactly on a snapshot date.
        on_date = service.resolve_universe(self.NAME, as_of=date(2017, 6, 1), registry=registry)
        assert on_date.symbols == ["AAA", "BBB", "DDD"]
        # Between snapshots → most recent snapshot at or before as_of.
        between = service.resolve_universe(self.NAME, as_of=date(2017, 3, 15), registry=registry)
        assert between.symbols == ["AAA", "BBB", "CCC"]
        # After the last snapshot → the last snapshot.
        after = service.resolve_universe(self.NAME, as_of=date(2017, 12, 29), registry=registry)
        assert after.symbols == ["AAA", "BBB", "DDD"]

    def test_before_first_snapshot_is_rejected(self, service: DataService, tmp_path: Path) -> None:
        from liq.data.universes import UniverseResolutionError

        self._write_snapshot(tmp_path)
        registry = self._write_registry_yaml(tmp_path)
        with pytest.raises(UniverseResolutionError, match="before first snapshot"):
            service.resolve_universe(self.NAME, as_of=date(2016, 12, 30), registry=registry)

    def test_missing_snapshot_parquet_is_fail_closed(
        self, service: DataService, tmp_path: Path
    ) -> None:
        from liq.data.universes import UniverseResolutionError

        registry = self._write_registry_yaml(tmp_path)  # YAML present, no parquet
        with pytest.raises(UniverseResolutionError, match="no point-in-time membership snapshot"):
            service.resolve_universe(self.NAME, as_of=date(2017, 2, 1), registry=registry)

    def test_registry_writes_canonical_declaration_text(self, tmp_path: Path) -> None:
        registry = UniverseRegistry(tmp_path)
        path = registry.save(self._definition())
        assert path == tmp_path / "reference" / "universes" / "sp500-pit.yaml"
        assert path.read_text(encoding="utf-8") == (
            "kind: composite\nname: sp500-pit\nspec:\n  id: sp500\n  source: snapshot\nversion: 1\n"
        )


# ----- real-data-root smoke (opt-in) -----------------------------------------


@pytest.mark.realdata
@pytest.mark.skipif(
    os.environ.get("RUN_REALDATA") != "1",
    reason="set RUN_REALDATA=1 to resolve sp500-pit against the real DATA_ROOT",
)
def test_real_root_sp500_pit_membership() -> None:
    """Read-only resolution of ``sp500-pit`` against the real data root.

    Default test runs MUST NOT touch the real data root; this check is
    gated on ``RUN_REALDATA=1`` so it stays opt-in. It reads reference
    data only (universe YAML + the per-id membership snapshot parquet) —
    never market-data bars. Probe dates sit inside the snapshot corpus's
    coverage (capped at 2024-12-31); the plausibility band brackets the
    S&P 500's actual seat count, which floats slightly above 500 because
    of multi-class listings.
    """
    from liq.data.settings import get_settings
    from liq.data.universes import UniverseRegistry as RealRegistry

    registry = RealRegistry(get_settings().data_root)
    service = DataService()
    for probe in (
        date(2017, 6, 19),
        date(2019, 7, 1),
        date(2021, 9, 20),
        date(2023, 6, 20),
        date(2024, 11, 26),
    ):
        resolved = service.resolve_universe("sp500-pit", as_of=probe, registry=registry)
        assert resolved.pit is True
        assert 490 <= len(resolved.symbols) <= 510
        assert "AAPL" in resolved.symbols

"""TDD tests for universe resolution + sources + registry."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from liq.data.universes import (
    ConstituentSource,
    InMemoryStubSource,
    ResolvedUniverse,
    UniverseConflictError,
    UniverseDefinition,
    UniverseKind,
    UniverseNotFoundError,
    UniverseRegistry,
    UniverseResolutionError,
    UniverseResolver,
)

# ----- constituent sources --------------------------------------------------


class TestInMemoryStubSource:
    def test_pit_flag_is_false(self) -> None:
        assert InMemoryStubSource().pit is False

    def test_members_returns_seeded_list(self) -> None:
        src = InMemoryStubSource({"SP500": ["AAPL", "MSFT"]})
        assert sorted(src.members(id="SP500", as_of=date(2025, 1, 2))) == ["AAPL", "MSFT"]

    def test_unknown_id_returns_empty(self) -> None:
        src = InMemoryStubSource({"SP500": ["A"]})
        assert src.members(id="UNKNOWN", as_of=date(2025, 1, 2)) == []

    def test_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryStubSource(), ConstituentSource)


# ----- resolver: explicit ---------------------------------------------------


class TestResolveExplicit:
    def test_passes_through_symbols_pit_true(self) -> None:
        d = UniverseDefinition(
            name="watch",
            version=1,
            kind=UniverseKind.EXPLICIT,
            spec={"symbols": ["MSFT", "AAPL"]},
        )
        r = UniverseResolver().resolve(d, date(2025, 1, 2))
        assert r.symbols == ["AAPL", "MSFT"]
        assert r.pit is True


# ----- resolver: filter -----------------------------------------------------


class _StaticReferenceData:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def as_of(self, when: date) -> pl.DataFrame:  # noqa: ARG002
        return self._df


class TestResolveFilter:
    @staticmethod
    def _ref_data() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "PENNY"],
                "price": [150.0, 380.0, 2.5],
                "dv20": [1.0e10, 1.5e10, 1.0e5],
                "security_type": ["common", "common", "common"],
                "exchange": ["NASDAQ", "NASDAQ", "OTC"],
            }
        )

    def test_filters_against_expression(self) -> None:
        ref = _StaticReferenceData(self._ref_data())
        resolver = UniverseResolver(reference_data=ref)
        d = UniverseDefinition(
            name="liquid",
            version=1,
            kind=UniverseKind.FILTER,
            spec={"expr": "(price > 5) & (dv20 > 10_000_000)"},
        )
        r = resolver.resolve(d, date(2025, 1, 2))
        assert r.symbols == ["AAPL", "MSFT"]
        assert r.pit is True

    def test_empty_reference_data_yields_empty_universe(self) -> None:
        ref = _StaticReferenceData(pl.DataFrame(schema={"symbol": pl.Utf8}))
        resolver = UniverseResolver(reference_data=ref)
        d = UniverseDefinition(
            name="x",
            version=1,
            kind=UniverseKind.FILTER,
            spec={"expr": "True"},
        )
        r = resolver.resolve(d, date(2025, 1, 2))
        assert r.symbols == []

    def test_missing_reference_data_raises(self) -> None:
        resolver = UniverseResolver()
        d = UniverseDefinition(
            name="x",
            version=1,
            kind=UniverseKind.FILTER,
            spec={"expr": "True"},
        )
        with pytest.raises(UniverseResolutionError, match="ReferenceData"):
            resolver.resolve(d, date(2025, 1, 2))

    def test_bad_expression_raises_with_useful_message(self) -> None:
        ref = _StaticReferenceData(self._ref_data())
        resolver = UniverseResolver(reference_data=ref)
        d = UniverseDefinition(
            name="x",
            version=1,
            kind=UniverseKind.FILTER,
            spec={"expr": "this is not python"},
        )
        with pytest.raises(UniverseResolutionError, match="compile"):
            resolver.resolve(d, date(2025, 1, 2))


# ----- resolver: composite ---------------------------------------------------


class TestResolveComposite:
    def test_inherits_pit_flag_from_source(self) -> None:
        d = UniverseDefinition(
            name="sp500",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "stub", "id": "SP500"},
        )
        src = InMemoryStubSource({"SP500": ["AAPL", "MSFT"]})
        r = UniverseResolver(constituent_source=src).resolve(d, date(2025, 1, 2))
        assert r.symbols == ["AAPL", "MSFT"]
        assert r.pit is False  # stub source is non-PIT

    def test_pit_source_yields_pit_universe(self) -> None:
        class _PITSource:
            pit = True

            def members(self, *, id: str, as_of: date):  # noqa: ARG002
                return ["AAPL", "MSFT"]

        d = UniverseDefinition(
            name="sp500",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "norgate", "id": "SP500"},
        )
        r = UniverseResolver(constituent_source=_PITSource()).resolve(d, date(2025, 1, 2))
        assert r.pit is True

    def test_missing_source_raises(self) -> None:
        d = UniverseDefinition(
            name="sp500",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "stub", "id": "SP500"},
        )
        with pytest.raises(UniverseResolutionError, match="ConstituentSource"):
            UniverseResolver().resolve(d, date(2025, 1, 2))


# ----- resolver: set_op -----------------------------------------------------


class TestResolveSetOp:
    @staticmethod
    def _resolver_with_inputs() -> UniverseResolver:
        return UniverseResolver(
            constituent_source=InMemoryStubSource({"SP500": ["AAPL", "MSFT", "TSLA"]}),
            named_universes={
                "sp500": UniverseDefinition(
                    name="sp500",
                    version=1,
                    kind=UniverseKind.COMPOSITE,
                    spec={"source": "stub", "id": "SP500"},
                ),
                "watch": UniverseDefinition(
                    name="watch",
                    version=1,
                    kind=UniverseKind.EXPLICIT,
                    spec={"symbols": ["AAPL", "GOOGL"]},
                ),
                "exclusion": UniverseDefinition(
                    name="exclusion",
                    version=1,
                    kind=UniverseKind.EXPLICIT,
                    spec={"symbols": ["TSLA"]},
                ),
            },
        )

    def test_union(self) -> None:
        d = UniverseDefinition(
            name="u",
            version=1,
            kind=UniverseKind.SET_OP,
            spec={"op": "union", "inputs": ["sp500", "watch"]},
        )
        r = self._resolver_with_inputs().resolve(d, date(2025, 1, 2))
        assert r.symbols == ["AAPL", "GOOGL", "MSFT", "TSLA"]
        assert r.pit is False  # SP500 via stub is non-PIT → set_op inherits

    def test_intersect(self) -> None:
        d = UniverseDefinition(
            name="u",
            version=1,
            kind=UniverseKind.SET_OP,
            spec={"op": "intersect", "inputs": ["sp500", "watch"]},
        )
        r = self._resolver_with_inputs().resolve(d, date(2025, 1, 2))
        assert r.symbols == ["AAPL"]

    def test_exclude(self) -> None:
        d = UniverseDefinition(
            name="u",
            version=1,
            kind=UniverseKind.SET_OP,
            spec={"op": "exclude", "inputs": ["sp500", "exclusion"]},
        )
        r = self._resolver_with_inputs().resolve(d, date(2025, 1, 2))
        assert r.symbols == ["AAPL", "MSFT"]

    def test_all_pit_inputs_yield_pit_universe(self) -> None:
        resolver = UniverseResolver(
            named_universes={
                "a": UniverseDefinition(
                    name="a", version=1, kind=UniverseKind.EXPLICIT, spec={"symbols": ["X"]}
                ),
                "b": UniverseDefinition(
                    name="b", version=1, kind=UniverseKind.EXPLICIT, spec={"symbols": ["Y"]}
                ),
            }
        )
        d = UniverseDefinition(
            name="u",
            version=1,
            kind=UniverseKind.SET_OP,
            spec={"op": "union", "inputs": ["a", "b"]},
        )
        r = resolver.resolve(d, date(2025, 1, 2))
        assert r.pit is True

    def test_unknown_input_raises(self) -> None:
        resolver = UniverseResolver()
        d = UniverseDefinition(
            name="u",
            version=1,
            kind=UniverseKind.SET_OP,
            spec={"op": "union", "inputs": ["missing"]},
        )
        with pytest.raises(UniverseResolutionError, match="unknown input"):
            resolver.resolve(d, date(2025, 1, 2))


# ----- registry --------------------------------------------------------------


class TestUniverseRegistry:
    def test_save_and_load_round_trips_explicit(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        d = UniverseDefinition(
            name="watch",
            version=1,
            kind=UniverseKind.EXPLICIT,
            spec={"symbols": ["AAPL"]},
        )
        path = reg.save(d)
        assert path.exists()
        round_tripped = reg.load("watch")
        assert round_tripped == d

    def test_save_round_trips_each_kind(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        kinds = [
            UniverseDefinition(
                name="explicit",
                version=1,
                kind=UniverseKind.EXPLICIT,
                spec={"symbols": ["AAPL"]},
            ),
            UniverseDefinition(
                name="filter",
                version=1,
                kind=UniverseKind.FILTER,
                spec={"expr": "price > 5"},
            ),
            UniverseDefinition(
                name="composite",
                version=1,
                kind=UniverseKind.COMPOSITE,
                spec={"source": "stub", "id": "SP500"},
            ),
            UniverseDefinition(
                name="set_op",
                version=1,
                kind=UniverseKind.SET_OP,
                spec={"op": "union", "inputs": ["a", "b"]},
            ),
        ]
        for d in kinds:
            reg.save(d)
        for d in kinds:
            assert reg.load(d.name) == d
        assert reg.list_names() == sorted(d.name for d in kinds)

    def test_idempotent_save_of_identical_definition(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        d = UniverseDefinition(
            name="x", version=1, kind=UniverseKind.EXPLICIT, spec={"symbols": ["A"]}
        )
        reg.save(d)
        reg.save(d)  # no exception
        assert reg.load("x") == d

    def test_collision_raises_without_overwrite(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        d1 = UniverseDefinition(
            name="x", version=1, kind=UniverseKind.EXPLICIT, spec={"symbols": ["A"]}
        )
        d2 = UniverseDefinition(
            name="x", version=2, kind=UniverseKind.EXPLICIT, spec={"symbols": ["B"]}
        )
        reg.save(d1)
        with pytest.raises(UniverseConflictError):
            reg.save(d2)

    def test_overwrite_replaces_persisted_definition(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        d1 = UniverseDefinition(
            name="x", version=1, kind=UniverseKind.EXPLICIT, spec={"symbols": ["A"]}
        )
        d2 = UniverseDefinition(
            name="x", version=2, kind=UniverseKind.EXPLICIT, spec={"symbols": ["B"]}
        )
        reg.save(d1)
        reg.save(d2, overwrite=True)
        assert reg.load("x") == d2

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        with pytest.raises(UniverseNotFoundError):
            reg.load("nope")

    def test_delete_returns_false_when_absent(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        assert reg.delete("nope") is False

    def test_delete_removes_existing(self, tmp_path: Path) -> None:
        reg = UniverseRegistry(tmp_path)
        d = UniverseDefinition(
            name="x", version=1, kind=UniverseKind.EXPLICIT, spec={"symbols": ["A"]}
        )
        reg.save(d)
        assert reg.delete("x") is True
        assert reg.list_names() == []

    def test_list_empty_when_no_dir(self, tmp_path: Path) -> None:
        assert UniverseRegistry(tmp_path / "nowhere").list_names() == []


class TestResolvedUniverseUsage:
    """One sanity check that a ``ResolvedUniverse`` instance is what the
    resolver returns (and not some other compatible shape)."""

    def test_resolver_returns_resolved_universe_instance(self) -> None:
        d = UniverseDefinition(
            name="x", version=1, kind=UniverseKind.EXPLICIT, spec={"symbols": ["A"]}
        )
        r = UniverseResolver().resolve(d, date(2025, 1, 2))
        assert isinstance(r, ResolvedUniverse)

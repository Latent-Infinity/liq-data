"""TDD tests for ``liq.data.universes`` domain types.

Locked contract:

* ``UniverseKind`` enum has exactly four members: ``explicit``,
  ``filter``, ``composite``, ``set_op``.
* ``UniverseDefinition`` is frozen, pydantic-validated, carries ``name``,
  ``version`` (int >= 1), ``kind``, and a ``spec`` whose shape is
  validated per kind.
* ``ResolvedUniverse`` is frozen, carries ``symbols`` (uppercase, sorted,
  unique), ``as_of`` (date), ``definition_version`` (int), ``pit`` (bool),
  and ``definition_name`` (str).
* PIT semantics — ``explicit`` / ``filter`` / ``set_op`` over PIT inputs
  are PIT by construction; ``composite`` inherits from its source.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError


class TestUniverseKind:
    def test_has_exactly_four_members(self) -> None:
        from liq.data.universes import UniverseKind

        assert {k.value for k in UniverseKind} == {
            "explicit",
            "filter",
            "composite",
            "set_op",
        }

    def test_kind_round_trips_through_value(self) -> None:
        from liq.data.universes import UniverseKind

        for k in UniverseKind:
            assert UniverseKind(k.value) is k


class TestUniverseDefinitionConstruction:
    def test_explicit_universe_holds_symbol_list(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        u = UniverseDefinition(
            name="watch",
            version=1,
            kind=UniverseKind.EXPLICIT,
            spec={"symbols": ["AAPL", "MSFT"]},
        )
        assert u.name == "watch"
        assert u.kind is UniverseKind.EXPLICIT
        assert u.spec["symbols"] == ["AAPL", "MSFT"]

    def test_filter_universe_holds_expression(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        u = UniverseDefinition(
            name="liquid",
            version=1,
            kind=UniverseKind.FILTER,
            spec={"expr": "price > 5 AND dv20 > 10_000_000"},
        )
        assert u.spec["expr"].startswith("price")

    def test_composite_universe_holds_source_id(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        u = UniverseDefinition(
            name="sp500",
            version=1,
            kind=UniverseKind.COMPOSITE,
            spec={"source": "stub", "id": "SP500"},
        )
        assert u.spec["source"] == "stub"
        assert u.spec["id"] == "SP500"

    def test_set_op_universe_holds_op_and_inputs(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        u = UniverseDefinition(
            name="combined",
            version=2,
            kind=UniverseKind.SET_OP,
            spec={
                "op": "union",
                "inputs": ["sp500", "watch"],
            },
        )
        assert u.spec["op"] == "union"
        assert u.spec["inputs"] == ["sp500", "watch"]

    def test_definition_is_frozen(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        u = UniverseDefinition(
            name="x",
            version=1,
            kind=UniverseKind.EXPLICIT,
            spec={"symbols": ["A"]},
        )
        with pytest.raises(ValidationError):
            u.name = "y"  # type: ignore[misc]


class TestUniverseDefinitionValidation:
    def test_version_must_be_positive(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        with pytest.raises(ValidationError):
            UniverseDefinition(
                name="x",
                version=0,
                kind=UniverseKind.EXPLICIT,
                spec={"symbols": ["A"]},
            )

    def test_name_must_be_non_empty(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        with pytest.raises(ValidationError):
            UniverseDefinition(
                name="",
                version=1,
                kind=UniverseKind.EXPLICIT,
                spec={"symbols": ["A"]},
            )

    def test_explicit_requires_symbols(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        with pytest.raises(ValueError, match="symbols"):
            UniverseDefinition(
                name="x",
                version=1,
                kind=UniverseKind.EXPLICIT,
                spec={},
            )

    def test_filter_requires_expr(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        with pytest.raises(ValueError, match="expr"):
            UniverseDefinition(
                name="x",
                version=1,
                kind=UniverseKind.FILTER,
                spec={"price": ">5"},
            )

    def test_composite_requires_source_and_id(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        with pytest.raises(ValueError):
            UniverseDefinition(
                name="x",
                version=1,
                kind=UniverseKind.COMPOSITE,
                spec={"source": "stub"},  # missing id
            )

    def test_set_op_requires_op_and_inputs(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        with pytest.raises(ValueError):
            UniverseDefinition(
                name="x",
                version=1,
                kind=UniverseKind.SET_OP,
                spec={"op": "union"},  # missing inputs
            )

    def test_set_op_rejects_unknown_operator(self) -> None:
        from liq.data.universes import UniverseDefinition, UniverseKind

        with pytest.raises(ValueError, match="op"):
            UniverseDefinition(
                name="x",
                version=1,
                kind=UniverseKind.SET_OP,
                spec={"op": "rotate", "inputs": ["a", "b"]},
            )


class TestResolvedUniverse:
    def test_carries_symbols_and_metadata(self) -> None:
        from liq.data.universes import ResolvedUniverse

        r = ResolvedUniverse(
            definition_name="watch",
            definition_version=1,
            symbols=["AAPL", "MSFT"],
            as_of=date(2025, 1, 2),
            pit=True,
        )
        assert r.symbols == ["AAPL", "MSFT"]
        assert r.pit is True
        assert r.as_of == date(2025, 1, 2)

    def test_symbols_normalized_uppercase_sorted_unique(self) -> None:
        """The resolver may emit duplicates / lowercase / unsorted; the
        type normalizes so consumers can rely on the canonical shape."""
        from liq.data.universes import ResolvedUniverse

        r = ResolvedUniverse(
            definition_name="watch",
            definition_version=1,
            symbols=["msft", "AAPL", "aapl"],
            as_of=date(2025, 1, 2),
            pit=True,
        )
        assert r.symbols == ["AAPL", "MSFT"]

    def test_is_frozen(self) -> None:
        from liq.data.universes import ResolvedUniverse

        r = ResolvedUniverse(
            definition_name="x",
            definition_version=1,
            symbols=["A"],
            as_of=date(2025, 1, 2),
            pit=True,
        )
        with pytest.raises(ValidationError):
            r.symbols = ["B"]  # type: ignore[misc]

    def test_empty_symbol_list_is_allowed(self) -> None:
        """An empty universe is legal (e.g. filter returned nothing); it's
        the sync layer that decides whether to warn."""
        from liq.data.universes import ResolvedUniverse

        r = ResolvedUniverse(
            definition_name="x",
            definition_version=1,
            symbols=[],
            as_of=date(2025, 1, 2),
            pit=True,
        )
        assert r.symbols == []

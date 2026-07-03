"""Universe definitions, resolution, and persistence.

A *universe* is a named, versioned object that names a set of symbols
of interest. Resolving it as-of a date returns a concrete symbol list
plus a point-in-time (PIT) flag the rest of the pipeline can act on
(e.g. sweep mode refuses non-PIT inputs to avoid survivorship bias).

Four kinds are supported. The kind dictates the spec shape:

* ``explicit`` — literal list of symbols. Always PIT.
* ``filter`` — a Polars expression over reference data (price, dollar
  volume, security type, exchange). PIT iff the reference data is PIT,
  which it is here because ``ReferenceData`` is keyed by ``as_of``.
* ``composite`` — index membership (e.g. ``SP500``) looked up via a
  :class:`ConstituentSource`. PIT iff the source claims to be PIT. The
  default :class:`InMemoryStubSource` is current-membership-only
  (``pit=False``); :class:`SnapshotConstituentSource` resolves historical
  full-composition snapshots with floor semantics (``pit=True``).
* ``set_op`` — ``union`` / ``intersect`` / ``exclude`` over other
  universe definitions. PIT iff every input is PIT.

Persistence lives in :class:`UniverseRegistry` — YAML files under
``{root}/reference/universes/{name}.yaml`` because these specs are
hand-edited.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ----- enums & spec keys -----------------------------------------------------


class UniverseKind(StrEnum):
    """The four supported universe kinds."""

    EXPLICIT = "explicit"
    FILTER = "filter"
    COMPOSITE = "composite"
    SET_OP = "set_op"


_VALID_SET_OPS: frozenset[str] = frozenset({"union", "intersect", "exclude"})


# ----- domain types ----------------------------------------------------------


class UniverseDefinition(BaseModel):
    """Frozen, validated description of a named universe.

    Subfield validation happens per-kind in :meth:`_validate_spec` — the
    spec dict's shape depends on which kind is selected, and the type
    system enforces just that one shape rather than admitting any dict.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    kind: UniverseKind
    spec: dict[str, Any]

    @model_validator(mode="after")
    def _validate_spec(self) -> UniverseDefinition:
        kind = self.kind
        spec = self.spec
        if kind is UniverseKind.EXPLICIT:
            if "symbols" not in spec or not isinstance(spec["symbols"], list):
                raise ValueError("explicit universe spec must contain a 'symbols' list")
        elif kind is UniverseKind.FILTER:
            if "expr" not in spec or not isinstance(spec["expr"], str):
                raise ValueError("filter universe spec must contain an 'expr' string")
        elif kind is UniverseKind.COMPOSITE:
            for k in ("source", "id"):
                if k not in spec or not isinstance(spec[k], str):
                    raise ValueError(f"composite universe spec must contain a {k!r} string")
        elif kind is UniverseKind.SET_OP:
            op = spec.get("op")
            inputs = spec.get("inputs")
            if op not in _VALID_SET_OPS:
                raise ValueError(
                    f"set_op universe spec 'op' must be one of {sorted(_VALID_SET_OPS)}"
                )
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("set_op universe spec must have a non-empty 'inputs' list")
        return self


class ResolvedUniverse(BaseModel):
    """Frozen result of resolving a definition as-of a date.

    ``symbols`` are normalized (uppercase, deduped, sorted) so consumers
    can rely on a canonical shape regardless of how the resolver
    produced them.
    """

    model_config = ConfigDict(frozen=True)

    definition_name: str
    definition_version: int
    symbols: list[str]
    as_of: date
    pit: bool

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: Sequence[str]) -> list[str]:
        seen: dict[str, None] = {}
        for s in value:
            seen.setdefault(str(s).upper(), None)
        return sorted(seen)


# ----- protocols -------------------------------------------------------------


@runtime_checkable
class ConstituentSource(Protocol):
    """Composite-universe resolver.

    Implementations look up membership for an index/grouping ``id``
    as-of a date. ``pit`` advertises whether the source can do
    point-in-time lookups (true historical membership) or only current
    membership (which silently excludes delisted / removed names and
    biases downstream analyses).
    """

    @property
    def pit(self) -> bool: ...

    def members(self, *, id: str, as_of: date) -> Sequence[str]: ...


class InMemoryStubSource:
    """Current-membership-only constituent source.

    Always returns ``pit=False`` so callers — particularly sweep mode —
    can refuse it before reading any bars (avoiding the survivorship
    bias trap). Backed by a literal ``{id: [symbols]}`` dict supplied
    at construction; tests inject directly.
    """

    pit: bool = False

    def __init__(self, mappings: dict[str, Sequence[str]] | None = None) -> None:
        self._mappings: dict[str, list[str]] = {
            id_: list(syms) for id_, syms in (mappings or {}).items()
        }

    def members(self, *, id: str, as_of: date) -> Sequence[str]:  # noqa: ARG002 — as_of unused for current-only
        return list(self._mappings.get(id, ()))


class SnapshotConstituentSource:
    """Point-in-time constituent source backed by full-composition snapshots.

    Each snapshot row is the complete index membership on its date;
    ``members`` resolves with floor semantics (the most recent snapshot at
    or before ``as_of``). ``pit=True`` because historical membership is
    reproduced as-of any covered date — asking for a date before the first
    snapshot is an error, never a silent current-membership fallback.
    """

    pit: bool = True

    def __init__(self, constituent_id: str, snapshots: pl.DataFrame) -> None:
        self._id = constituent_id
        frame = snapshots.sort("date")
        self._dates: list[date] = frame["date"].to_list()
        self._tickers: list[list[str]] = [
            sorted({str(t).upper() for t in row}) for row in frame["tickers"].to_list()
        ]

    @classmethod
    def from_parquet(cls, constituent_id: str, path: Path | str) -> SnapshotConstituentSource:
        """Load snapshots (``date``, ``tickers`` list column) from parquet."""
        return cls(constituent_id, pl.read_parquet(path))

    def members(self, *, id: str, as_of: date) -> Sequence[str]:
        if id != self._id:
            raise UniverseResolutionError(
                f"unknown constituent id {id!r}; this source serves {self._id!r}"
            )
        position = bisect_right(self._dates, as_of) - 1
        if position < 0:
            raise UniverseResolutionError(
                f"as_of {as_of} is before first snapshot {self._dates[0]}"
            )
        return list(self._tickers[position])


@runtime_checkable
class ReferenceData(Protocol):
    """Reference data adapter for filter-kind universes.

    The single method returns one Polars row per symbol with at least
    the columns ``symbol``, ``price``, ``dollar_volume_20d``,
    ``security_type``, ``exchange`` — the predicates the spec ``expr``
    is allowed to reference. Implementations decide where the data
    comes from (EQUS.SUMMARY snapshot today; cached parquet tomorrow).
    """

    def as_of(self, when: date) -> pl.DataFrame: ...


# ----- resolver --------------------------------------------------------------


class UniverseResolver:
    """Resolves a :class:`UniverseDefinition` to a :class:`ResolvedUniverse`.

    Dependency-injected with a constituent source (composite kind) and a
    reference-data adapter (filter kind). ``set_op`` resolves its inputs
    recursively via a small lookup dict so callers don't have to manage
    the dependency graph manually.

    PIT propagation rules:

    * ``explicit`` is PIT by construction.
    * ``filter`` is PIT iff the reference-data adapter is keyed by
      ``as_of`` (always true given the protocol shape).
    * ``composite`` inherits from the constituent source.
    * ``set_op`` is the logical AND of its inputs' PIT flags.
    """

    def __init__(
        self,
        *,
        constituent_source: ConstituentSource | None = None,
        reference_data: ReferenceData | None = None,
        named_universes: dict[str, UniverseDefinition] | None = None,
    ) -> None:
        self._constituent_source = constituent_source
        self._reference_data = reference_data
        self._named = dict(named_universes or {})

    def register(self, definition: UniverseDefinition) -> None:
        """Make a definition available for ``set_op`` input lookups."""
        self._named[definition.name] = definition

    def resolve(self, definition: UniverseDefinition, as_of: date) -> ResolvedUniverse:
        symbols, pit = self._resolve_inner(definition, as_of)
        return ResolvedUniverse(
            definition_name=definition.name,
            definition_version=definition.version,
            symbols=list(symbols),
            as_of=as_of,
            pit=pit,
        )

    # ----- internal --------------------------------------------------

    def _resolve_inner(self, definition: UniverseDefinition, as_of: date) -> tuple[list[str], bool]:
        kind = definition.kind
        if kind is UniverseKind.EXPLICIT:
            return list(definition.spec["symbols"]), True
        if kind is UniverseKind.FILTER:
            return self._resolve_filter(definition, as_of)
        if kind is UniverseKind.COMPOSITE:
            return self._resolve_composite(definition, as_of)
        if kind is UniverseKind.SET_OP:
            return self._resolve_set_op(definition, as_of)
        raise ValueError(f"unknown universe kind: {kind!r}")  # pragma: no cover

    def _resolve_filter(
        self, definition: UniverseDefinition, as_of: date
    ) -> tuple[list[str], bool]:
        if self._reference_data is None:
            raise UniverseResolutionError("filter universe requires a ReferenceData adapter")
        df = self._reference_data.as_of(as_of)
        if df.is_empty():
            return [], True
        expr_str = definition.spec["expr"]
        # The expression is a constrained Polars boolean expression
        # over the documented columns. We sandbox eval to a minimal
        # namespace so accidental name lookups can't reach into the
        # process.
        namespace = {
            "pl": pl,
            "col": pl.col,
            **{c: pl.col(c) for c in df.columns},
        }
        try:
            predicate = eval(expr_str, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            raise UniverseResolutionError(
                f"filter universe spec.expr failed to compile: {exc}"
            ) from exc
        filtered = df.filter(predicate)
        if "symbol" not in filtered.columns:
            raise UniverseResolutionError(
                "filter universe reference data must include a 'symbol' column"
            )
        return filtered["symbol"].to_list(), True

    def _resolve_composite(
        self, definition: UniverseDefinition, as_of: date
    ) -> tuple[list[str], bool]:
        if self._constituent_source is None:
            raise UniverseResolutionError("composite universe requires a ConstituentSource")
        id_ = definition.spec["id"]
        symbols = list(self._constituent_source.members(id=id_, as_of=as_of))
        return symbols, bool(self._constituent_source.pit)

    def _resolve_set_op(
        self, definition: UniverseDefinition, as_of: date
    ) -> tuple[list[str], bool]:
        op = definition.spec["op"]
        input_names = definition.spec["inputs"]

        resolved: list[tuple[set[str], bool]] = []
        for name in input_names:
            inner = self._named.get(name)
            if inner is None:
                raise UniverseResolutionError(
                    f"set_op universe references unknown input universe {name!r}"
                )
            symbols, inner_pit = self._resolve_inner(inner, as_of)
            resolved.append(({s.upper() for s in symbols}, inner_pit))

        first_set, _ = resolved[0]
        result_set = set(first_set)
        for sym_set, _ in resolved[1:]:
            if op == "union":
                result_set |= sym_set
            elif op == "intersect":
                result_set &= sym_set
            elif op == "exclude":
                result_set -= sym_set
            else:  # pragma: no cover — caught by validator
                raise ValueError(op)

        all_pit = all(p for _, p in resolved)
        return sorted(result_set), all_pit


# ----- exceptions ------------------------------------------------------------


class UniverseError(Exception):
    """Base class for universe-related failures."""


class UniverseResolutionError(UniverseError):
    """Resolution failed (missing dependency, bad spec, etc.)."""


class UniverseNotFoundError(UniverseError):
    """Registry lookup didn't find the named universe."""


class UniverseConflictError(UniverseError):
    """Tried to save a universe whose version already exists on disk
    without an explicit overwrite."""


# ----- registry --------------------------------------------------------------


class UniverseRegistry:
    """File-backed CRUD for universe definitions.

    Files live under ``{root}/reference/universes/{name}.yaml``. YAML
    chosen over parquet because humans hand-edit these specs; YAML
    round-trips through the pydantic model with no information loss.
    """

    SUBDIR = Path("reference") / "universes"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def directory(self) -> Path:
        return self._root / self.SUBDIR

    def _path_for(self, name: str) -> Path:
        return self.directory / f"{name}.yaml"

    def list_names(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.yaml"))

    def load(self, name: str) -> UniverseDefinition:
        path = self._path_for(name)
        if not path.exists():
            raise UniverseNotFoundError(f"universe {name!r} not found at {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return UniverseDefinition.model_validate(data)

    def save(
        self,
        definition: UniverseDefinition,
        *,
        overwrite: bool = False,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path_for(definition.name)
        if path.exists() and not overwrite:
            existing = self.load(definition.name)
            if existing == definition:
                return path
            raise UniverseConflictError(
                f"universe {definition.name!r} already exists at {path}; "
                f"pass overwrite=True to replace"
            )
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.safe_dump(definition.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    def delete(self, name: str) -> bool:
        path = self._path_for(name)
        if not path.exists():
            return False
        path.unlink()
        return True


__all__ = [
    "ConstituentSource",
    "InMemoryStubSource",
    "ReferenceData",
    "ResolvedUniverse",
    "SnapshotConstituentSource",
    "UniverseConflictError",
    "UniverseDefinition",
    "UniverseError",
    "UniverseKind",
    "UniverseNotFoundError",
    "UniverseRegistry",
    "UniverseResolutionError",
    "UniverseResolver",
]

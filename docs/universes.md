# Universes

A *universe* is a named, versioned object that names a set of symbols
of interest. Resolving a universe as-of a date yields a concrete symbol
list plus a point-in-time (PIT) flag that downstream code (e.g. sweep
mode) acts on.

The four kinds, the spec they carry, and how each resolves:

| Kind | Spec keys | Resolution | PIT? |
| --- | --- | --- | --- |
| `explicit` | `symbols: list[str]` | Pass-through after normalization (uppercase, sorted, deduped) | Always `True` |
| `filter` | `expr: str` (Polars boolean expression) | Evaluated against a `ReferenceData.as_of(date)` frame; surviving `symbol` column wins | `True` (reference data is keyed by `as_of`) |
| `composite` | `source: str`, `id: str` | `ConstituentSource.members(id=, as_of=)` | Inherits `source.pit` — the default `InMemoryStubSource` is `False` |
| `set_op` | `op: str` (`union` / `intersect` / `exclude`), `inputs: list[str]` | Recursively resolve every input, then apply the op | AND of all inputs' `pit` flags |

## Storage

Definitions live as YAML under
`{DATA_ROOT}/reference/universes/{name}.yaml`. YAML was chosen over
parquet because these specs are hand-edited.

```yaml
name: sp500_liquid
version: 3
kind: set_op
spec:
  op: intersect
  inputs:
    - sp500
    - liquid
```

`UniverseRegistry` provides `save` / `load` / `list_names` / `delete`.
Saving a definition whose name already exists raises
`UniverseConflictError` unless `overwrite=True` is set.

## Resolution

```python
from datetime import date
from liq.data.universes import UniverseRegistry, UniverseResolver

reg = UniverseRegistry("/var/data")
definition = reg.load("sp500_liquid")
resolver = UniverseResolver(
    named_universes={n: reg.load(n) for n in reg.list_names()},
    constituent_source=...,   # required only for composite kind
    reference_data=...,       # required only for filter kind
)
resolved = resolver.resolve(definition, as_of=date(2025, 1, 2))
print(resolved.symbols, resolved.pit)
```

The resolver normalizes symbols (uppercase, sorted, deduped) so
consumers can rely on a canonical shape regardless of how a particular
input emitted them.

## PIT semantics

PIT propagation is the only non-obvious resolution rule. Survivorship
bias — silently dropping delisted / acquired names — is the trap, and
it concentrates exactly in the population where high-magnitude moves
live, so it would poison downstream analysis.

* `explicit` is PIT by construction (the operator picked the list).
* `filter` is PIT because the reference adapter is keyed by `as_of`.
* `composite` inherits its `ConstituentSource.pit` flag. The bundled
  `InMemoryStubSource` is current-membership-only, so any composite
  universe resolved through it is `pit=False`. A real PIT vendor
  (Norgate, etc.) lights `pit=True` once wired in.
* `set_op` is the logical AND of its inputs: union/intersect/exclude
  over a PIT and a non-PIT input is non-PIT.

Sweep mode in `liq-scan` refuses to read against a non-PIT universe —
see the requirements doc for the exact rule.

When `sync(...)` resolves a non-PIT universe it emits a `pit_warning`
log event (catalog in `docs/logging.md`) and proceeds — the sync runs,
the operator sees the warning, and any downstream sweep refuses the
result before reading bars.

## Symbology renames

A single raw symbol can map to different venue instrument ids across a
backfill window (e.g., `FB → META` in 2022). Each Databento fetch
persists its symbology table as append-only rows keyed by
`(raw_symbol, instrument_id, valid_from, valid_to)`, so multi-row
mappings round-trip through one parquet location
(`reference/databento/symbology`).

`DatabentoProvider.resolve_symbol_for_date(symbol, as_of)` reads the
persisted table and returns the instrument id whose validity window
contains `as_of`. Returns `None` if no row covers the date — callers
should fail loud rather than picking the wrong id silently.

## CLI

```
liq-data universe create --name watch --kind explicit --symbols AAPL,MSFT
liq-data universe create --name liquid --kind filter --expr "(price > 5) & (dv20 > 10_000_000)"
liq-data universe create --name combo --kind set_op --op intersect --inputs sp500,watch
liq-data universe list
liq-data universe resolve watch --as-of 2025-01-02
liq-data universe delete watch
```

Every write path emits a single line of JSON to stdout for piping.
Logs go to stderr.

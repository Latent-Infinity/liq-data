# AGENT_SUMMARY — liq-data

Final summary for `liq-data`'s contribution to the
[`liq-scan-plan`](../liq-docs/plans/liq-scan-plan.md). `liq-data` carried
the heaviest share of this plan's work (DatabentoProvider, universes,
coverage manifest, sync).

## Status

| Field | Value |
| --- | --- |
| Plan | `liq-scan-plan` |
| Visibility | Public (MIT) |
| Final phase | F+1 |
| Unresolved blockers | _None._ |

## What this plan added to liq-data

### Phase 1 — DatabentoProvider MVP (`0b1fd10`)

Explicit-symbol historical bar fetches, EQUS dataset routing, integer-to-
Decimal normalization, `get_range` vs. batch routing, symbology sidecar
persistence, structured logging, settings/export integration, CLI support
(positional and `--provider/--symbols/--output json` forms).

### Phase 1H — DatabentoProvider hardening (`b38cc9b`)

Tenacity-backed retry, 429/5xx/transport error translation, resumable
batch-download markers, schema-mismatch errors, deterministic local 1m
rollups, fetch-time aggregate cross-checks, Databento-specific
exception/logging docs.

### Phase 2 — UniverseDefinition + coverage manifest + sync (`6802a6f`)

Pydantic `UniverseDefinition` (symbol-list / filter / composite / set-op
kinds), YAML registry persistence, explicit/filter/composite/set-op
resolution, parquet coverage manifests with `start_ts`/`end_ts`,
transaction rollback, registry-backed sync planning, force-refresh,
universe CLI commands, direct PyYAML declaration.

### Phase 2H — Universe and manifest hardening (`adb9475`)

Resumability after per-symbol provider failure, zero-row missing-bar
manifest claims, non-PIT warning logs, structured sync event constants,
manifest rollback events, file-lock serialization, symbology as-of lookup
across rename windows.

### Phase 3 dependency landing (`c5c1304`)

Lock metadata updated to include `liq-store`'s direct `duckdb>=1.0`
dependency.

### Phase 4 — Shared calendar/universe (`d2144fd`)

NYSE window helpers centralized; `DataService.resolve_universe(...)`
exposed for scanner execution.

### Phase 4H — Calendar edge support (`23fc2c5`)

Extended-hours minute grid, half-day coverage, DST tests landed in
`liq-data` to keep calendar source resolution in one helper.

## Verify-final evidence

- `artifacts/phase-F/verify.txt` — `ruff format --check && ruff check &&
  ty check && pytest -q` all green; project coverage **90.79 %**.
- Live Databento smoke skipped (no `RUN_DATABENTO=1` set).

## Cost ledger

| Date | Event | Cost | Authorization |
| --- | --- | --- | --- |
| 2026-06-12 | Phase 2H live SPY pilot | `$0.000688` est. (`61,600` billable bytes for SPY 2024-06-03 → 2024-06-04, 1100 rows) | Operator in-session |

## Per-phase commits (summary)

| Phase | Commit |
| --- | --- |
| 0 | `35caa96` (ruff drift, deps, contract stubs, ty cleanup 30 → 0) |
| 1 | `0b1fd10` |
| 1H | `b38cc9b` |
| 2 | `6802a6f` |
| 2H | `adb9475` |
| 3 | `c5c1304` |
| 4 | `d2144fd` |
| 4H | `23fc2c5` |
| F | _(this commit)_ |
| F+1 | _(this commit)_ |

## Out-of-scope items confirmed absent

No live broker integration, no signal/risk logic, no scheduler/cadence
ownership.

## Follow-on work (named for forward planning)

- Norgate PIT constituent data integration (required for composite
  production universes per Requirements R-1).
- Production-scale Databento backfill: operator-gated per plan §6 item 13.

# AGENT_STATE.md — liq-data

State tracked per implementation plan that touches this repo.

## Plan: `liq-features-canonical-risk-variance-impl-plan` (v1.2)

| Phase | Capability | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Calendar gap helpers — `closed_hours_between` + `classify_gap` + docs | ready-for-review | `artifacts/phase-0/` |

## Plan: `liq-scan-plan`

| Field | Value |
| --- | --- |
| Plan | [`../liq-docs/plans/liq-scan-plan.md`](../liq-docs/plans/liq-scan-plan.md) |
| Requirements | [`../liq-docs/requirements/liq-scan-requirements.md`](../liq-docs/requirements/liq-scan-requirements.md) |
| Execution branch | `main` (single-developer model) |
| Last updated | 2026-06-21 |

| Phase | Status | Verify | Commit | Notes |
| --- | --- | --- | --- | --- |
| 0 — Foundation | done | green | `35caa96` | Ruff drift cleanup, deps, 3 xfail contract stubs, ty cleanup (30 → 0) |
| 1 — MVP DatabentoProvider | done | green | `0b1fd10` | `docs/databento-provider.md` |
| 1H — Harden DatabentoProvider | done | green | `b38cc9b` | `docs/exceptions.md`, `docs/logging.md`; live SPY pilot deferred to 2H |
| 2 — MVP Universe + manifest + sync | done | green | `6802a6f` | `docs/universes.md`, `docs/coverage-manifest.md` |
| 2H — Harden universes + manifest | done | green | `adb9475` | Live SPY pilot at `$0.000688` est. (1,100 rows, 61,600 billable bytes) |
| 3 — Phase 3 lock metadata | done | green | `c5c1304` | `uv.lock` for `duckdb>=1.0` transitive |
| 4 — MVP shared calendar | done | green | `d2144fd` | `src/liq/data/calendar.py`, `DataService.resolve_universe` |
| 4H — Calendar edge support | done | green | `23fc2c5` | Extended-hours grid, half-day, DST tests |
| 5 / 5H — Sweep | n/a |  |  | Owned by liq-scan |
| F — Docs polish | done | green | _(this commit)_ | `artifacts/phase-F/verify.txt`; coverage 90.79 % |
| F+1 — Final verification | done | green | _(this commit)_ | `AGENT_SUMMARY.md` |

Open follow-ups: _None_ for `liq-scan-plan`. Blocked entries: _None._

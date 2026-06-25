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

## Plan: `breakout-following-strategy-impl-plan` (v0.2.8)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| B0 | Required upstream enhancement (impl plan §1.5.1) landed: `DataService.estimate_databento_cost(universe, *, start, end, timeframe, dataset, registry=None) -> dict` — wraps Databento's non-billable `metadata.get_cost(...) -> float` USD and `metadata.get_billable_size(...) -> int` bytes endpoints (verified against `databento` 0.79.0 surface, constructor signature `Historical(key=...)` per `src/liq/data/providers/databento.py:1170`). Derives `schema="ohlcv-1m"` from `timeframe="1m"`; refuses any other timeframe. Refuses missing `DATABENTO_API_KEY`, empty universes, inverted date ranges. Returns dict with keys `{billable_bytes, estimated_cost_usd, dataset, schema, symbols, start, end, provider_request_id}` without triggering any billable download. Universe-registration row for `ai-chip-liquid-30-riskvar` already present at `data/financial_data/reference/universes/ai-chip-liquid-30-riskvar.yaml` version 1 (30 symbols); no additional registration needed. | ready-for-review | `src/liq/data/service.py:763` (`estimate_databento_cost`); `tests/test_estimate_databento_cost.py` (20 new tests, all green). Workspace-standard verify (`uv run pytest && uv run ruff check src tests && uv run ruff format --check src tests && uv run ty check src`) green: 735 passed + 1 pre-existing skip. |
| B3 | Cross-cohort acquisition throughput pre-deliverable landed. `DataService.sync(...)` now accepts keyword-only `max_workers: int = 1`; the default keeps the sequential code path, while `max_workers > 1` runs per-symbol work in a bounded `ThreadPoolExecutor` under the existing universe file lock. Per-symbol manifest transactions still roll back failed symbols and preserve completed symbols for resumable sync; rate-limiter entry is serialized so parallel workers do not race mutable limiter state. | ready-for-review | `src/liq/data/service.py`; `tests/test_sync_max_workers.py` (13 pytest cases covering sequential default, bounded concurrency, serialized rate-limiter entry, parallel rollback, and lock semantics). Verified with `uv run pytest tests/test_sync_max_workers.py -q --no-cov`, `uv run ruff check src/liq/data/service.py tests/test_sync_max_workers.py`, `uv run ruff format --check src/liq/data/service.py tests/test_sync_max_workers.py`, and `uv run ty check src/liq/data/service.py tests/test_sync_max_workers.py`. |

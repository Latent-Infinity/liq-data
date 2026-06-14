# `liq-data` structured log event catalog

Every provider in `liq-data` emits structured log records via the
standard `logging` module. Fields are attached via the `extra=` kwarg
so they land as queryable attributes on `LogRecord` rather than being
baked into the message string.

This file catalogs the Databento provider's events. Other providers
follow the same shape — see their respective module docstrings.

## Correlation key: `sync_run_id`

Every fetch carries a freshly generated `sync_run_id` (a UUID string).
All events emitted while servicing that fetch — including retries —
share the same value. Use it to reconstruct the full attempt sequence
for a single user-facing call:

```bash
jq 'select(.sync_run_id == "abc123…")' liq-data.log
```

## `databento_fetch`

Emitted once per successful `fetch_bars` invocation, after records are
materialized and symbology is persisted (or skipped).

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"databento_fetch"` |
| `provider` | `str` | Always `"databento"` |
| `dataset` | `str` | Routed Databento dataset code (e.g. `EQUS.MINI`) |
| `symbols_count` | `int` | Number of symbols in the request (currently always 1) |
| `start` | `str` | ISO date |
| `end` | `str` | ISO date |
| `request_kind` | `str` | `"get_range"` or `"batch"` |
| `bytes_in` | `int` | Best-effort byte estimate from the response |
| `duration_ms` | `int` | Wall-clock latency of the entire call (including any retries) |
| `sync_run_id` | `str` | UUID; see § Correlation |

Example:

```
2025-01-02T14:30:01Z INFO  databento fetch EQUS.MINI/AAPL 390 records
  event=databento_fetch provider=databento dataset=EQUS.MINI
  symbols_count=1 start=2025-01-02 end=2025-01-02
  request_kind=get_range bytes_in=12480 duration_ms=842
  sync_run_id=8a3b4c5d-…
```

## `databento_retry`

Emitted once **per retry attempt** by `_call_with_retry` after a
`DatabentoTransientError` is caught and before sleeping for backoff.
Not emitted for the initial attempt (it has no preceding failure to
log) and not emitted after the final attempt (the exception propagates
instead).

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"databento_retry"` |
| `provider` | `str` | Always `"databento"` |
| `sync_run_id` | `str` | UUID — **matches the parent `databento_fetch`** for the eventual success |
| `dataset` | `str` | Routed Databento dataset code |
| `symbol` | `str` | The symbol the request was for |
| `request_kind` | `str` | `"get_range"` or `"batch"` |
| `attempt` | `int` | 1-indexed attempt number that just failed |
| `max_attempts` | `int` | Total budget; provider attribute `max_retry_attempts` |
| `backoff_s` | `float` | Seconds the provider will sleep before retrying |
| `error_type` | `str` | Concrete `DatabentoTransientError` subclass name |
| `error_message` | `str` | `str(exc)` of the caught error |

Example: a 429 followed by a successful retry produces this sequence
(same `sync_run_id` throughout):

```
2025-01-02T14:30:00Z INFO databento retry EQUS.MINI attempt=1
  event=databento_retry sync_run_id=8a3b4c5d-… attempt=1
  max_attempts=3 backoff_s=2.5 error_type=DatabentoRateLimitError
  error_message="429 Too Many Requests"

2025-01-02T14:30:03Z INFO databento fetch EQUS.MINI/AAPL 390 records
  event=databento_fetch sync_run_id=8a3b4c5d-… duration_ms=3041
  request_kind=get_range bytes_in=12480
```

The `duration_ms` on the `databento_fetch` event covers the entire
wall-clock window including the backoff sleep — so it answers "how
long did the user wait" rather than "how long did the venue take."

## `DataService.sync` events

`DataService.sync(universe, ...)` emits a small evergreen catalog of
structured events to the `liq.data.service` logger. Every event in one
sync invocation shares a single `sync_run_id` (UUID) so the operator
can reconstruct the run from log output. Event-name constants live in
`liq.data.sync_events` — string mismatches between emitter and parser
are not possible.

### `sync_started`

Emitted exactly once per `sync(...)` call, before any work runs.
Paired with `sync_completed` at the end so an operator can confirm
the sync did not silently exit between universe resolution and
manifest commit.

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"sync_started"` |
| `sync_run_id` | `str` | UUID; see § Reconstructing a session |
| `universe`, `version`, `kind` | — | Definition metadata |
| `provider`, `dataset`, `timeframe` | `str` | Routed request shape |
| `start`, `end` | `str` | ISO dates of the requested window |
| `force_refresh` | `bool` | True iff the call passed `force_refresh=True` |

### `universe_resolved`

Emitted once per `sync(...)` call, immediately after the resolver
produces the symbol list and before any provider work.

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"universe_resolved"` |
| `sync_run_id` | `str` | UUID; see § Reconstructing a session |
| `universe` | `str` | Definition name |
| `version` | `int` | Definition version |
| `kind` | `str` | One of `explicit / filter / composite / set_op` |
| `provider`, `dataset`, `timeframe` | `str` | Routed request shape |
| `symbols_count` | `int` | Number of resolved symbols |
| `as_of` | `str` | ISO date the universe was resolved as-of |
| `pit` | `bool` | True iff the resolved universe is point-in-time |

### `pit_warning`

Emitted at level `WARNING` only when `pit=False` (e.g., a composite
universe resolved through the in-memory stub source). Sync proceeds —
downstream sweep mode refuses non-PIT input separately.

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"pit_warning"` |
| `sync_run_id` | `str` | Matches the parent `universe_resolved` |
| `universe`, `version`, `kind` | — | As above |
| `provider`, `dataset`, `timeframe` | `str` | Routed request shape |
| `reason` | `str` | Human-readable explanation (e.g., `"constituent source did not advertise PIT membership"`) |

### `manifest_gap_detected`

Emitted once per symbol that needs work — zero gaps means no event.

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"manifest_gap_detected"` |
| `sync_run_id` | `str` | Matches the parent `universe_resolved` |
| `symbol` | `str` | The symbol being planned |
| `gaps_count` | `int` | Number of uncovered ranges in the requested window |

### `manifest_range_appended`

Emitted once per successful provider fetch + manifest commit (inside
the per-symbol transaction). A zero-row fetch still emits this event —
the manifest claim is recorded so the next sync does NOT re-bill the
venue (see FR-4 / `docs/coverage-manifest.md`).

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"manifest_range_appended"` |
| `sync_run_id` | `str` | Matches the parent `universe_resolved` |
| `symbol` | `str` | The symbol whose range landed |
| `start`, `end` | `str` | ISO timestamps of the half-open `[start, end)` range |
| `rows` | `int` | Bars returned by the provider (can be 0 for thin symbols) |

### `manifest_rollback`

Emitted at level `ERROR` when a per-symbol transaction exits via
exception — the manifest claim is reverted before the exception
propagates. Earlier symbols' commits are not affected, so a re-run
picks up where the failure occurred.

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"manifest_rollback"` |
| `sync_run_id` | `str` | Matches the parent `universe_resolved` |
| `symbol` | `str` | The symbol whose transaction rolled back |
| `error_type` | `str` | Concrete exception class name |
| `error_message` | `str` | `str(exc)` of the caught error |

### `symbol_started` / `symbol_completed` / `symbol_failed`

Per-symbol progress envelope. `symbol_started` fires once per symbol
that has at least one gap (fully-covered symbols are silent so noise
stays low on incremental runs). `symbol_completed` fires after the
transaction commits with the row total; `symbol_failed` fires once
at level `ERROR` paired with `manifest_rollback`.

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | One of `symbol_started` / `symbol_completed` / `symbol_failed` |
| `sync_run_id` | `str` | Matches the parent `sync_started` |
| `symbol` | `str` | The per-symbol identifier |
| `gaps_count` | `int` | (`started` / `completed`) Number of uncovered ranges planned |
| `rows` | `int` | (`completed`) Total rows fetched for the symbol |
| `error_type`, `error_message` | `str` | (`failed`) See `manifest_rollback` |

### `sync_completed`

Emitted exactly once per `sync(...)` call after the lock is released.

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"sync_completed"` |
| `sync_run_id` | `str` | Matches the parent `sync_started` |
| `symbols`, `api_calls`, `rows_fetched`, `manifest_gaps` | `int` | Final tally — mirrors the return value of `sync(...)` |

## Where batch artifacts land

`DatabentoProvider` produces two kinds of artifacts per batch job —
one durable, one ephemeral:

* **Resume marker** — `{batch_jobs_dir}/{signature}.json`. Default
  location is `${TMPDIR}/liq-data/databento-batch-jobs/`. Markers
  survive process restarts so an interrupted job is never
  re-submitted (which would re-bill the venue).
* **Downloaded `.dbn.zst` files + SDK siblings** — staged inside a
  per-call `tempfile.TemporaryDirectory()` with the
  `liq-data-databento-` prefix. The provider materializes the records
  + symbology into memory **inside** the context manager, then
  returns; the `with` block exit deletes the staging directory on
  both success and failure. Operators never see these files; they
  are an implementation detail of the batch fetch.

Records flow through the normal liq-data ingestion path
(`_extract_records` → `_records_to_dataframe` →
`ParquetStore.write` under `DATA_ROOT/{provider}/{symbol}/bars/...`)
exactly like the `get_range` happy path. By the time
`fetch_bars` returns, the staging dir is gone and only the parquet
bars + manifest range remain.

The `.gitignore` includes `/EQUS-*/` as a backstop against earlier
code that wrote into `Path.cwd()`; current code does not need it
but the guard prevents accidents during rollback.

## Databento batch events

`DatabentoProvider` emits one event per state transition while a
batch job is in flight. Each carries `provider="databento"` and the
batch job's `job_id`.

| Event | Level | When |
| --- | --- | --- |
| `batch_submitted` | INFO | A new batch job was submitted — emitted only when no resumable marker existed for the request signature. Carries `dataset`, `symbol`, `start`, `end`, `job_id`. |
| `batch_resumed` | INFO | A prior marker was found and is being reused. No re-submission, no re-billing. Carries `dataset`, `symbol`, `job_id`. |
| `batch_polling` | INFO | Once per poll tick while the job is still in a non-terminal state. Carries `job_id`, `state`. |
| `batch_download_started` | INFO | The job hit a terminal state; the download is about to start. Carries `job_id`, `output_dir`. |

The order is `batch_submitted` (or `batch_resumed`) → 0..N
`batch_polling` → `batch_download_started` → the existing
`databento_fetch` event (success) or the raised exception
(`DatabentoError`).

## Batch orchestration mode

`liq-data sync` defaults to the original serial path. Providers that
implement the generic batch lifecycle (`submit_batch_bars`,
`poll_batch_bars`, `fetch_completed_batch_bars`) can opt into bounded
batch orchestration:

```
$ liq-data sync sp500 \
  --start 2024-01-02 \
  --end 2024-12-31 \
  --provider databento \
  --timeframe 1m \
  --dataset EQUS.MINI \
  --orchestration batch \
  --max-in-flight 4 \
  --verbose
```

The service keeps up to `--max-in-flight` provider batch jobs active.
It still downloads and writes completed jobs through the normal
manifest transaction path one gap at a time. This means a process
interruption can leave provider-side jobs in flight, but each provider
must persist durable resume markers before submission is considered
successful. On restart, the same request resumes existing jobs instead
of re-submitting them.

The same event stream is used in both modes. In batch orchestration
operators should expect several `batch_submitted` / `batch_resumed`
events before the first `batch_download_started` event.

### Per-mode envelope fields

| Field | Serial `sync(...)` | `sync_batch(...)` |
| --- | --- | --- |
| `sync_started.orchestration` | _absent_ | `"batch"` |
| `sync_started.max_in_flight` | _absent_ | int |
| Return value `orchestration` | _absent_ | `"batch"` |
| Return value `max_in_flight` | _absent_ | int |

### Failure semantics in `sync_batch`

`symbol_failed` is emitted with a `stage` field naming the lifecycle
step that raised:

| `stage` value | Where the failure happened |
| --- | --- |
| `"submit"` | `submit_batch_bars(...)` raised; no manifest claim existed yet, so no rollback is needed for this symbol |
| `"poll"` | `poll_batch_bars(...)` raised on an in-flight job; the orchestrator re-raises so the operator can decide policy |
| _(no `stage` field)_ | The download/manifest transaction raised; paired with `manifest_rollback` |

In every case the run aborts after emitting `symbol_failed`; previously
committed symbols stay persisted and an `sync_completed` event is
**not** emitted (consistent with serial `sync(...)`).

### Rate-limiter accounting

`sync_batch` charges the configured rate limiter for every
`submit_batch_bars` and `fetch_completed_batch_bars` call. **Polling
is exempt** — it is a status check, not a billable data request.
If polling were charged, a high `--max-in-flight` value would burn
the per-minute budget on poll ticks alone and starve the actual
submits + downloads. The provider's `poll_batch_bars` is responsible
for its own retry on transient SDK errors.

## CLI heartbeat

`liq-data sync --verbose` installs a stderr handler on the
`liq.data` parent logger that prints one short line per event.
Stdout still carries the final JSON report so pipes that consume the
report shape are unaffected.

```
$ liq-data sync sp500 --start 2024-01-02 --end 2024-12-31 --verbose
[sync_started]
[universe_resolved]
[symbol_started] symbol=AAPL
[batch_submitted] symbol=AAPL job_id=db-abc123
[batch_polling] job_id=db-abc123 state=queued
[batch_polling] job_id=db-abc123 state=processing
[batch_download_started] job_id=db-abc123
[symbol_completed] symbol=AAPL
…
[sync_completed]
{"symbols": 503, "api_calls": 503, "rows_fetched": 49_000_000, …}
```

## Reconstructing a session

The `sync_run_id` correlation key makes session reconstruction a single
filter:

```python
import json, sys
target = sys.argv[1]
for line in sys.stdin:
    rec = json.loads(line)
    if rec.get("sync_run_id") == target:
        print(rec["event"], rec.get("attempt", ""), rec.get("error_type", ""))
```

```
$ cat liq-data.log | python session.py 8a3b4c5d-…
databento_retry 1 DatabentoRateLimitError
databento_retry 2 DatabentoTransientError
databento_fetch
```

…tells the operator that the fetch took two retries (one for rate
limit, one for a transient blip) before succeeding.

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

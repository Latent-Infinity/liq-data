# `DatabentoProvider`

US-equities historical bars via Databento — 1m via `EQUS.MINI`, 1d
via `EQUS.SUMMARY`. Implements the `BaseProvider` contract with
deterministic integer-to-Decimal normalization, batch-vs-sync routing
by date span, and symbology persistence. Universes, the coverage
manifest, and `sync(universe)` are not part of this module.

## Dataset routing

| Asset class | Timeframe | Databento dataset | Source map |
| --- | --- | --- | --- |
| `equity` | `1m` | `EQUS.MINI` | `DATASET_ROUTING` |
| `equity` | `1d` | `EQUS.SUMMARY` | `DATASET_ROUTING` |

The map lives in `src/liq/data/providers/databento.py` as a literal
`dict[tuple[str, str], str]`. Adding futures (`GLBX.MDP3`) or crypto
later is a one-line entry, not a structural change. Requesting an
unmapped `(asset_class, timeframe)` raises `ProviderError`.

The `(timeframe → schema name)` mapping is a sibling literal:

| Timeframe | Databento schema |
| --- | --- |
| `1m` | `ohlcv-1m` |
| `1d` | `ohlcv-1d` |

## Batch vs. sync routing

`fetch_bars(symbol, start, end, timeframe)` picks one of two Databento
endpoints based on the request span:

```
span_days = (end - start).days + 1

span_days <  DATABENTO_BATCH_THRESHOLD_DAYS  → timeseries.get_range  (sync stream)
span_days >= DATABENTO_BATCH_THRESHOLD_DAYS  → batch.submit_job  → download
```

**Threshold default: 14 days.** Constructor argument `batch_threshold_days`
lets tests and operators flip it.

### Why 14?

* Below ~14 calendar days, `timeseries.get_range` returns in under one
  request RTT for a single equity symbol — the latency of submitting +
  polling a batch job dominates.
* At or above ~14 days, Databento's batch download is materially cheaper
  (per the published pricing) and faster overall for a single sequential
  consumer.
* The exact crossover is empirical and Databento may tune it; the value
  is config not constant for that reason.

### Batch recovery

Batch requests are keyed by `(dataset, symbol, schema, start, end)` and
persist a small marker under the provider's batch-job cache directory.
If a download fails after submission, the next invocation with the same
request signature resumes the existing Databento job instead of
submitting a duplicate job. The marker is deleted only after a clean
download. Terminal failed job states remove the marker and raise
`DatabentoError` so a later run can submit a fresh job.

## Decimal precision

Databento ships OHLCV prices as `int64` scaled by `1e9` (DBN-internal
"q9" convention). The provider converts via

```python
(Decimal(value_q9) / DATABENTO_PRICE_SCALE).quantize(Decimal("1e-8"))
```

— an exact rational divide, no float intermediate, quantized to the
`PRICE_DTYPE` scale of 8 decimal places. The hypothesis test in
`tests/providers/test_databento.py::TestHypothesisIntegerDecimalRoundTrip`
fuzzes this round-trip across 50 random q9 values per run.

Timestamps arrive as `int64` UNIX nanoseconds and are normalized to
`Datetime("us", "UTC")` (microsecond precision, UTC-aware).

## Symbology persistence

Every fetch carries a symbology mapping — a
`{raw_symbol: [{instrument_id, valid_from, valid_to}, ...]}` dict —
because Databento's instrument IDs can drift across history (renames,
ticker reuses). The provider appends these rows to `liq-store` under

```
reference/databento/symbology
```

as a long-format Polars frame with columns
`raw_symbol | instrument_id | valid_from | valid_to`. Downstream queries
through raw symbol consult this table to find the right `instrument_id`
for an as-of window.

Symbology persistence is best-effort: if the provider was constructed
without a `store=`, it computes the frame but doesn't persist. If the
fetch returned an empty symbology dict, no write happens.

## Structured log events

Every successful fetch emits one `INFO` record on the
`liq.data.providers.databento` logger with the following queryable
fields (set via `extra=`):

| Field | Type | Description |
| --- | --- | --- |
| `event` | `str` | Always `"databento_fetch"` |
| `provider` | `str` | Always `"databento"` |
| `dataset` | `str` | Routed Databento dataset code |
| `symbols_count` | `int` | Number of symbols in the request |
| `start` | `str` | ISO date |
| `end` | `str` | ISO date |
| `request_kind` | `str` | `"get_range"` or `"batch"` |
| `bytes_in` | `int` | Best-effort inbound byte count from the SDK response, or a deterministic row-based estimate when unavailable |
| `duration_ms` | `int` | Wall-clock latency of the underlying call |
| `sync_run_id` | `str` | Per-fetch correlation ID |

These fields land on `LogRecord` attributes; tests assert their
presence (see `tests/providers/test_databento.py::TestLogging`).

Transient retries emit `event="databento_retry"` before sleeping for
backoff. Retry logs include the same `sync_run_id` as the eventual
`databento_fetch` success, plus `attempt`, `max_attempts`, `backoff_s`,
`error_type`, and `error_message`.

## Retries and schema checks

The provider uses `tenacity` for bounded retry. Databento-specific
`DatabentoTransientError` and `DatabentoRateLimitError` are retryable.
Raw SDK exceptions are also translated when they expose HTTP status
metadata:

| Input failure | Provider behavior |
| --- | --- |
| HTTP `429` | Converts to `DatabentoRateLimitError`, honors `Retry-After` when present, and retries |
| HTTP `5xx` | Converts to `DatabentoTransientError` and retries with exponential backoff |
| Transport timeout / connection failure | Converts to `DatabentoTransientError` and retries |
| Other provider error | Propagates without retry |

Response stores that expose a schema must match the requested schema.
A mismatch raises `DatabentoSchemaError` before records are materialized.

## Local aggregation and cross-checks

`aggregate_bars(df_1m, timeframe=...)` deterministically rolls 1m bars
to `5m`, `15m`, `30m`, `1h`, or `1d` without issuing another Databento
request. The aggregation uses wall-clock bucket boundaries:

| Field | Rule |
| --- | --- |
| `open` | first value in bucket |
| `high` | max value in bucket |
| `low` | min value in bucket |
| `close` | last value in bucket |
| `volume` | sum |

`validate_aggregate(symbol, date)` compares locally aggregated 1m data
against `EQUS.SUMMARY` for the same date. Price fields must be within
`0.001 %` of the local/venue close midpoint; volume must match exactly.
Callers that want validation as part of ingest can use
`fetch_bars(..., timeframe="1m", validate_aggregate=True)`, which
reuses the already fetched 1m frame and only performs the daily summary
comparison request.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `DATABENTO_API_KEY` | Required for any real-Databento call. Absent → `create_databento_provider` raises `ValueError`. |
| `RUN_DATABENTO` | Opt-in switch for the live-API smoke test (`tests/providers/test_databento.py::test_real_databento_smoke`). Default test runs MUST NOT call Databento. |

## Network policy

The default `pytest` suite uses an in-process fake client and never
touches the network. Only `tests/providers/test_databento.py::test_real_databento_smoke`
calls the real API, and that test is `@pytest.mark.databento`-gated +
checks `RUN_DATABENTO=1` before doing anything.

## Out of scope

* **Universes, manifest, `sync(universe)`** — live in `liq.data.universes`
  and `liq.data.manifest` (separate work stream).
* **Multi-symbol single-call fetch** — the universe `sync(...)` path
  consolidates fetches across a universe; the per-symbol API here
  stays single-symbol for predictability and quota accounting.
* **Real-time / Live API** — historical only; intentionally not in
  scope.

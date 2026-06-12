# `liq-data` exception catalog

All custom exceptions live in `liq.data.exceptions` (base hierarchy) and
the per-provider modules under `liq.data.providers.*`. This file lists
the load-bearing ones a caller should know about.

## Base hierarchy

```
DataError                       — root for every liq-data error
├── ProviderError               — anything venue-specific
│   ├── RateLimitError          — generic 429 (not all providers raise this)
│   ├── AuthenticationError     — bad / expired credentials
│   ├── ProviderUnavailableError — venue health issue
│   └── DatabentoError          — Databento-specific (see below)
├── ValidationError
│   └── SchemaValidationError   — local schema check failed
├── ConfigurationError          — bad config / missing env var
└── DataQualityError            — ingested data failed integrity check
    └── AggregateCrossCheckError — 1m-aggregate vs 1d mismatch
```

## Databento-specific errors

Everything under `liq.data.providers.databento.DatabentoError`:

| Exception | When raised | Retry-eligible? | Recovery |
| --- | --- | --- | --- |
| `DatabentoError` | Base; concrete subclasses below | n/a | — |
| `DatabentoTransientError` | 5xx, connection blips, transient network failures from the SDK | **Yes** — the provider's `_call_with_retry` honors `max_retry_attempts` (default 3) with exponential backoff. If `exc.retry_after` is set, that value is honored verbatim. | Auto-retried by the provider. If retries are exhausted the exception propagates with its last-attempt context. |
| `DatabentoRateLimitError` | Subclass of `DatabentoTransientError`. Raised for 429 responses; carries the `Retry-After` hint when the venue supplied one. | **Yes** — same retry path as transient; venue hint overrides exponential backoff. | Auto-retried; consider lowering request frequency if it recurs. |
| `DatabentoSchemaError` | Response store carries a schema string that doesn't match the requested schema (e.g. asked for `ohlcv-1m`, got `ohlcv-1h`). | **No** — retrying would return the same mismatched store. | Investigate dataset routing or schema request; check the Databento console for recent venue changes. |
| `AggregateCrossCheckError` | `validate_aggregate(symbol, date)` found a 1m-aggregate vs `EQUS.SUMMARY` 1d mismatch. Price fields exceeded 0.001 % of midrange close, or volume did not match exactly. | **No** — data quality concern, not a transient failure. | Inspect `exc.diffs` for the field-by-field drift; consider re-ingesting with `force_refresh` or escalating to the venue. |

## How callers should handle them

```python
from liq.data.providers.databento import (
    DatabentoProvider,
    DatabentoRateLimitError,
    DatabentoSchemaError,
    AggregateCrossCheckError,
)

try:
    df = provider.fetch_bars("AAPL", start, end, timeframe="1m")
except DatabentoSchemaError as exc:
    # Permanent — check dataset routing.
    log.error("schema mismatch", extra={"error": str(exc)})
    raise
except DatabentoRateLimitError:
    # Already auto-retried; budget is genuinely exhausted.
    log.warning("rate-limited after all retries; backing off")
    raise
```

For cross-checks:

```python
try:
    report = provider.validate_aggregate("AAPL", date(2025, 1, 2))
except AggregateCrossCheckError as exc:
    for field, diff in exc.diffs.items():
        log.warning(
            "1m vs 1d drift",
            extra={"field": field, "relative": str(diff["relative"])},
        )
    raise
```

The `AggregateCrossCheckError.diffs` shape is
`{field: {"local": Decimal, "venue": Decimal, "absolute": Decimal, "allowed": Decimal}}`
for every OHLCV component that exceeded the tolerance. Fields within
tolerance are not present.

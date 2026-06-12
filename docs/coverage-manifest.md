# Coverage manifest

The coverage manifest is the idempotency mechanism for
`DataService.sync(universe, ...)`. Before any provider call we ask the
manifest: *which ranges of (provider, dataset, timeframe, symbol) have
we already fetched?* Anything covered is skipped.

The alternative — inspecting stored bars — is ambiguous: a missing
minute could be "not fetched" *or* "no trades occurred" (thin symbols,
halts, partial sessions). The manifest removes that ambiguity by
recording acquisition intent explicitly.

## Storage layout

One parquet file per (provider, dataset, timeframe, symbol):

```
{DATA_ROOT}/metadata/{provider}/{dataset}/{timeframe}/{symbol}/manifest.parquet
```

Each row is one contiguous covered range, half-open `[start_ts, end_ts)`:

| Column | Type | Notes |
| --- | --- | --- |
| `start_ts` | `Datetime("us", "UTC")` | Inclusive |
| `end_ts` | `Datetime("us", "UTC")` | Exclusive |
| `fetched_at` | `Datetime("us", "UTC")` | Latest fetch contributor's wall-clock |
| `batch_job_id` | `Utf8` (nullable) | Set when a single batch job covered the range; nulled out when ranges from multiple sources merge |

Ranges are kept sorted and merged on insert — adjacent or overlapping
inputs collapse to a single row.

## Gap detection

`manifest.gaps(start, end)` returns the uncovered slices of
`[start, end)` not already in the manifest:

* Empty manifest → `[(start, end)]` (one big gap)
* Fully covered window → `[]`
* Partial coverage → the requested window minus the covered slices,
  sorted ascending

Gaps are exactly what `DataService.sync(...)` fetches; ranges already
in the manifest stay un-touched.

## Transactional updates

The only safe way to mutate is `manifest.transaction()`:

```python
with manifest.transaction() as txn:
    for gap_start, gap_end in gaps:
        df = provider.fetch_bars(...)
        store.write(...)
        txn.record(CoverageRange(start=gap_start, end=gap_end, fetched_at=...))
```

On normal exit the body's `record(...)` calls commit in one parquet
flush. If the body raises, the manifest rolls back to its
pre-transaction snapshot — the partial writes that already landed in
`liq-store` stay there, but the manifest never claims to cover them,
so the next sync re-fetches.

Re-using a transaction object after exit raises
`ManifestRollbackError`.

## Operational notes

* `force_refresh=True` on sync skips gap detection and re-fetches the
  full window. Use sparingly — this re-bills the venue.
* The manifest tracks *acquisition intent*, not data integrity.
  Validating that ingested rows actually round-trip to the venue's
  daily aggregate is `DatabentoProvider.validate_aggregate(...)`'s
  job.
* No automatic pruning; the manifest only grows as universes do.
  Range merging keeps it bounded in practice.

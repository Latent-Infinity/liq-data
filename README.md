# liq-data
Part of the Latent Infinity Quant (LIQ) ecosystem, `liq-data` handles data acquisition from external sources. It fetches raw market data from various providers, normalizes it to the shared `liq-core`, and persists it via `liq-store`.

## Architecture: liq-store Integration

**All data access in liq-data goes through liq-store exclusively.** This is a critical architectural requirement that ensures:

- **Single Source of Truth**: All data reads/writes use `ParquetStore` from liq-store
- **Automatic Deduplication**: Append operations merge data by timestamp automatically
- **Consistent Storage Keys**: Bars use `provider/<symbol>/bars/<timeframe>` (e.g., `oanda/EUR_USD/bars/1m`). Quotes/fundamentals/corp actions use `provider/<symbol>/quotes|fundamentals|corp_actions`. Use `liq.store.key_builder` helpers to avoid drift. Higher timeframes are aggregated from 1m when missing.
- **Supported rollups**: Standard frames (1m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 12h, 1d, plus any whole-minute frame) are aggregated from 1m on demand and cached back to the store, aligned to wall-clock boundaries.

| Requested timeframe | Source used           | Notes                          |
|---------------------|-----------------------|--------------------------------|
| 1m                  | 1m                    | Pass-through                   |
| 5m / 15m / 30m      | 1m                    | Aggregated and cached          |
| 1h / 2h / 4h / 8h   | 1m                    | Aggregated and cached          |
| 12h / 1d / 2d       | 1m                    | Aggregated and cached          |
| other N m/h/d       | 1m                    | Any whole-minute frame supported|
- **SOLID/DRY Principles**: No direct parquet access (`pl.read_parquet`, `df.write_parquet`) in the codebase

### Usage Pattern

```python
from liq.data.service import DataService

ds = DataService()
# Fetch and store bars
ds.fetch("oanda", "EUR_USD", start, end, timeframe="1m")
# Load from store
df = ds.load("oanda", "EUR_USD", "1m")
# Backfill missing gaps
df = ds.backfill("oanda", "EUR_USD", start, end, timeframe="1m")
```

### Research lockbox guard

Research reads must declare a `purpose` and `arm_id`; `DataService.load` then
enforces the campaign lockbox ledger (`liq.data.lockbox`) before touching any
data and appends every permitted read to `<data_root>/lockbox_usage_log.jsonl`:

```python
df = ds.load(
    "oanda", "EUR_USD", "1m",
    start=date(2020, 1, 1), end=date(2023, 12, 31),
    purpose="discovery", arm_id="idea_05a",
)
```

Reads inside a program-lockbox period raise `LockboxViolationError` unless the
human-only `final_portfolio_review=True` flag is passed; an arm gets exactly
one validation-period use per dataset (`ValidationReuseError` on reuse);
`purpose="dev_smoke"` is allowed anywhere but tagged in the log and is never
research evidence. Reads without a declared purpose are not checked or logged
and can never be cited as research evidence.

For final provenance reconciliation, query citable windows with both identity
dimensions: `guard.guarded_windows(arm_id=arm_id, dataset=dataset)`. The result
contains only research-purpose reads for that exact dataset; `dev_smoke` reads
are deliberately excluded.

### Point-in-time composite universes

Composite universe definitions must name their constituent source. A declared
`source: snapshot` resolves `{data_root}/reference/universes/snapshots/{id}.parquet`
as full-composition `date` / `tickers` snapshots and fails closed if the file or
schema is invalid. The current-membership-only, non-PIT source is available only
through an explicit `source: stub` declaration; it is never an implicit fallback.

### SEC 8-K event metadata

`SECEdgarProvider.fetch_8k_events` returns exact-token matches for registered
8-K item families together with acceptance, accession, and primary-document
metadata. The returned lookup symbol is not point-in-time identity evidence;
research consumers must verify CIK-to-symbol history separately.

`SECEdgarProvider.fetch_accession_metadata` reads one official complete-
submission SGML file and returns the filing's Inline-XBRL trading-symbol facts,
exact document types, and an `EX-99.1` attachment flag. This evidence is bound
to the filing accession and never falls back to the current ticker map. Missing
or multiple filing symbols remain explicit for downstream exclusion and
coverage reporting.

For identity-readiness audits, `ticker_candidates` preserves every exact
current SEC ticker/CIK/name association instead of overwriting duplicate
tickers, and `filing_index` returns deterministic accession references from the
current plus archived submissions indexes. The current ticker file is candidate
metadata only; an accession-bound trading-symbol fact is still required for
filing-date evidence, and neither API infers a continuous identity interval.

`SECEdgarProvider.get_corporate_actions` conservatively extracts explicitly
disclosed cash dividends and stock splits from candidate 8-K documents. It
retains accession, document, and evidence hashes; never infers an ex-date; and
skips SEC-typed ZIP, GRAPHIC, and EXCEL packaging attachments. Potentially
action-bearing PDF exhibits are fetched from their standalone SEC archive URL,
verified by PDF magic, and converted with `pdftotext -layout`; install Poppler's
`pdftotext` executable before using this method. Missing, malformed, empty, or
timed-out PDF extraction and unsupported textual markup fail closed. Declared
research calls through `DataService.fetch_corporate_actions` share the arm's
lockbox governance.

`resolve_edgar_event_clock` converts a timezone-aware SEC acceptance timestamp
into the XNYS session bucket, public/event timestamps, reaction-window end, and
latency decision timestamps. Before-open, after-close, weekend, and holiday
events are re-anchored to the applicable next regular-session open.

### Anti-patterns to Avoid

Do NOT use direct parquet access:
```python
# BAD - direct parquet access
df = pl.read_parquet(path)
df.write_parquet(path)
pl.scan_parquet(path)

# GOOD - use liq-store
store = get_store()
df = store.read(key)
store.write(key, df, mode="append")
```

### TradeStation Auth Code Flow

TradeStation uses Auth0 authorization code flow to obtain refresh tokens.
Configure these in `.env` (see `.env.example`), then:

```bash
# Step 1: Generate authorization URL
uv run python -m liq.data.cli tradestation-auth-url

# Step 2: Exchange the returned code for a refresh token
uv run python -m liq.data.cli tradestation-exchange-code YOUR_CODE
```

If TradeStation rotates refresh tokens, liq-data can persist updates back to `.env`
when `TRADESTATION_PERSIST_REFRESH_TOKEN=true` (default).

## QA CLI

Run bar-level QA checks on stored data or files:
```bash
# Fetch bars via CLI
liq-data fetch oanda EUR_USD --start 2024-01-01 --end 2024-01-31 --timeframe 1m

# Backfill missing data
liq-data backfill oanda EUR_USD --start 2024-01-01 --end 2024-01-31 --timeframe 1m

# Detect gaps
liq-data gaps --provider oanda --symbol EUR_USD --timeframe 1m --expected-minutes 1

# Validate provider credentials
liq-data validate-credentials oanda

# Load aggregated bars (1h from 1m on-the-fly)
python - <<'PY'
from liq.data.service import DataService
ds = DataService()
df_1h = ds.load("oanda", "EUR_USD", "1h")
print(df_1h.head())
PY
```

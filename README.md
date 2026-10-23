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
# Every read declares what it is for. Ingestion and diagnostics are not
# research, so they declare dev_smoke and name the component in arm_id.
ds.fetch(
    "oanda", "EUR_USD", start, end, timeframe="1m",
    purpose="dev_smoke", arm_id="my_ingest_component",
)
df = ds.load(
    "oanda", "EUR_USD", "1m", start, end,
    purpose="dev_smoke", arm_id="my_ingest_component",
)
df = ds.backfill(
    "oanda", "EUR_USD", start, end, timeframe="1m",
    purpose="dev_smoke", arm_id="my_ingest_component",
)
```

### Historical bid/ask candles

`DataService.fetch_bid_ask_bars("oanda", symbol, start, end, timeframe="1m",
purpose=purpose, arm_id=arm_id)` requests OANDA historical `price=BA` candles
through the research-period guard. Both UTC date bounds are inclusive. The
result contains `timestamp`, `bid_open/high/low/close`, `ask_open/high/low/close`,
and `volume`, with decimal prices. Incomplete candles are omitted; malformed
complete candles are refused, not repaired. Missing timestamps remain missing.

This retrieval-only path does not persist to the midpoint bar store. Existing
`fetch_bars` behavior remains midpoint-based. Bid/ask sides already contain the
quoted spread: do not add it again when computing returns from those sides.
Historical candles use OANDA's base-price group and do not establish
account-specific executable prices, slippage, or financing. Research source
admission remains separate from API access and period-guard checks.

### Research lockbox guard

`DataService.preflight_stored_timestamps` inspects the timestamp footer declarations
of an exact SHA-256-pinned file inventory after the research guard admits the
requested date scope. Supply positive file, byte and row-group budgets. It refuses
missing or inconsistent statistics, unknown encodings, unexpected inventory paths
and row-group extrema outside the requested dates. Separate footer usage records
are written to `timestamp_metadata_usage.jsonl`. Metadata bounds are writer
declarations; successful preflight does not authorize page reads or prove that
the writer's statistics match the observations.
For immutable stores, `DataService(research_log_root=...)` places both research
guard and footer usage logs in a separate audit directory. The default remains
the data root; the source paths and guard rules are unchanged.

`DataService.audit_stored_timestamps` summarizes an admitted stored snapshot using
only the `timestamp` column. Supply explicit dates, purpose/arm and positive
`batch_size`/`max_rows` budgets. It uses guarded batches, never contacts a provider
or aggregates missing bars, and returns raw duplicate counts, UTC-date counts and
anomaly codes. UTC dates do not prove exchange-session identity or original row
ordering. A budget refusal yields no partial summary. Missing stored bars that
would need aggregation raise `StoredBarsRequiredError`, a `ValueError` subtype.
When the reader yields no batches, its schema is unverified and the audit raises
`SchemaValidationError`; this also refuses valid empty or fully filtered sources
whose schema the iterator does not expose. It never certifies a zero-row source
by inventing an empty UTC schema.

This API requires an independently admitted, physically bounded snapshot. The
storage reader enumerates every partition under a key before date filtering;
dates alone therefore do not establish physical source isolation. Historical
source use still needs its own pinned inputs and research authorization.

`audit_pinned_timestamps` audits the complete physical timestamp domain of an
exact key inventory against pinned footer declarations. It guards access before
reading bytes and projects timestamps from the same verified buffer used for
footer inspection. It refuses undeclared dates, duplicate timestamps, duplicate
UTC dates in daily keys, budget overruns, and declaration mismatches. It preserves
source order and reports separate descent counts within groups, between groups,
and between files. Completion requires exhausting every projection and writing
the usage record; a partial read never returns a completed audit. This API does
not itself authorize historical source access.

The guard is **default-deny**. Every entry point that can return market values
-- `load`, `iter_batches`, `fetch`, `fetch_corporate_actions`,
`fetch_adjustment_factors`, `probe_bar_coverage`, `backfill`, and the derived
`validate` / `info` / `stats` / `gaps` -- requires a `purpose` and `arm_id`. A
read that declares neither raises `UndeclaredReadError` before any store or
provider is touched. A declared read is checked against the campaign lockbox
ledger (`liq.data.lockbox`) and appended to
`<data_root>/lockbox_usage_log.jsonl`:

```python
df = ds.load(
    "oanda", "EUR_USD", "1m",
    start=date(2020, 1, 1), end=date(2023, 12, 31),
    purpose="discovery", arm_id="idea_05a",
)
```

`UndeclaredReadError` is deliberately **not** a `LockboxViolationError` and does
not inherit from it: "you did not say what this read is for" and "the fold
boundaries refuse this read" are different facts, and `except
LockboxViolationError` must not swallow the first. `UnclassifiedProviderError`
is the third fact -- "nobody classified this provider" -- and is likewise
unrelated to either by inheritance.

`AlpacaEventBasisDailyRequest` uses a separate
`CLASSICAL_RANKER_EVENT_BASIS_FIXTURE_LEDGER_V1` for the fixed sixteen raw/all
event slots. It preserves the raw-only announcement request and logs the two
adjustment modes as distinct datasets. This current-vintage arithmetic source
does not establish total return, action consistency or historical availability;
capture and numerical use require their separately frozen scopes.

#### Providers outside the ledger

A read whose provider `resolve_dataset` does not map is one of two things, and
they are no longer the same thing:

- **Ungoverned** -- the provider is in `liq.data.lockbox.UNGOVERNED_PROVIDERS`,
  an allow-list carrying a stated reason per entry (`fred`, `polygon`). The read
  is served *and logged*, with `dataset: null`, `governed: false`, the provider
  string and a stable `ungoverned:<provider>` identity. A null dataset matches
  no dataset name, so the entry can never supply citable coverage through
  `guarded_windows`; `guard.ungoverned_windows(arm_id=..., provider=...)`
  returns it for honest provenance attribution instead.
- **Unclassified** -- everything else. A research-purpose read raises
  `UnclassifiedProviderError` before any store or provider is touched. A
  `dev_smoke` read is served and logged, so ingestion, fixture builds and
  diagnostics keep working, and a genuinely misspelled provider still gets
  `_get_provider`'s more specific "Unknown provider" message.

Adding an allow-list entry is a governance act, not a code cleanup.

#### One venue, one storage key

The storage key is built from the caller's provider argument; the venue is chosen
by settings. Nothing joined the two, so a `BINANCE_USE_US=true` fetch under the
`binance` key wrote Binance.US bars into a key named for Binance.com and nothing
downstream could tell -- stored bars carry no venue provenance.

`liq.data.lockbox.STORAGE_KEY_VENUES` records, per key, which
`MarketDataProvider.name` it is declared to hold; every key without an entry
defaults to its own name. Every path that writes provider bars under a
caller-chosen key -- `fetch`, `backfill`, `sync`, `sync_batch` -- compares the
constructed provider against that declaration before contacting the provider and
raises `StorageVenueMismatchError` on disagreement. The one recorded entry is `binance`,
declared to hold `binance_us` bars because that is what the tree was measured to
contain -- so a truthful Binance.US ingest passes and a Binance.com ingest under
the same key is refused as market mixing.

`binance` and `binance_us` resolve to the same governed dataset (`binance_spot`);
they are one venue, so there is one key. **Do not copy or alias bars between
provider keys** -- that makes the same minutes reachable under two fold
governances. Evidence and reasoning:
`liq-docs/plans/oracle-binance-us-market-identity-2026-09-24.md`.

Reads inside a program-lockbox period raise `LockboxViolationError` unless the
human-only `final_portfolio_review=True` flag is passed; an arm gets exactly
one validation-period use per dataset (`ValidationReuseError` on reuse);
`purpose="dev_smoke"` is allowed anywhere but tagged in the log and is never
research evidence.

Research reads must be bounded -- an unbounded read cannot be checked against a
fold window at all. An unbounded read is admissible only for `dev_smoke`, and
its ledger entry records `start: null`, `end: null` and `unbounded: true`
rather than a window the read did not have. `validate`, `info`, `stats` and
`gaps` read the whole stored series and are therefore `dev_smoke`-only.

For final provenance reconciliation, query citable windows with both identity
dimensions: `guard.guarded_windows(arm_id=arm_id, dataset=dataset)`. The result
contains only research-purpose reads for that exact dataset; `dev_smoke` reads
are deliberately excluded.

`unreachable_datasets(ledger)` reports the opposite disagreement: datasets a
ledger declares that no `resolve_dataset` call can return, so the governance
written for them governs nothing. It is non-empty today (`binance_perp` in the
campaign ledger, `fred_macro` in the FCE ledger) and the reachability test is
marked `xfail` with the owed routing decision as its reason.

### Point-in-time composite universes

Composite universe definitions must name their constituent source. A declared
`source: snapshot` resolves `{data_root}/reference/universes/snapshots/{id}.parquet`
as full-composition `date` / `tickers` snapshots and fails closed if the file or
schema is invalid. The current-membership-only, non-PIT source is available only
through an explicit `source: stub` declaration; it is never an implicit fallback.

`SP500MembershipProvider.capture_wikipedia_revision_reference(cutoff)` requests
one page's revision metadata at or before a whole-second UTC cutoff. It returns
unparsed bytes below a 64 KiB ceiling, without redirects, retries, article content
or continuation requests. The authorized caller must retain those bytes before
parsing and supervise the whole-attempt deadline. A revision locator does not
establish historical index membership or security identity.

`liq.data.wikipedia_revision.parse_revision_metadata(raw, cutoff=...)` validates
an already retained metadata response and returns a scalar revision locator.
It refuses duplicate JSON keys, ambiguous page identity, suppression, extra
fields and clocks after the cutoff. The provider-declared SHA-1 is preserved
without claiming content verification; continuation is reported but never followed.

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
`filing_clock_index` is the fail-closed companion for session-based research:
it includes the official SEC acceptance timestamp and rejects a selected filing
whose clock is missing or malformed.

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

### Coinbase Derivatives contract metadata

Coinbase Exchange spot and Coinbase Derivatives Exchange (CDE/FairX) use
different hosts and credentials. `CoinbaseDerivativesProvider` is intentionally
read-only and currently exposes only authenticated contract identity from the
CDE instruments endpoint. Configure its separate DCC credentials with
`COINBASE_DERIVATIVES_API_KEY`, `COINBASE_DERIVATIVES_API_SECRET`, and
`COINBASE_DERIVATIVES_PASSPHRASE`. It has no funding-history, price, order, or
account method; those remain outside the interface until the applicable
research dataset/fold contract is human-frozen.

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
df_1h = ds.load("oanda", "EUR_USD", "1h", purpose="dev_smoke", arm_id="readme_example")
print(df_1h.head())
PY
```

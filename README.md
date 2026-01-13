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
python -m liq.data.cli tradestation-auth-url

# Step 2: Exchange the returned code for a refresh token
python -m liq.data.cli tradestation-exchange-code YOUR_CODE
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

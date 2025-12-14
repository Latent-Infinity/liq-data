# liq-data
Part of the Latent Infinity Quant (LIQ) ecosystem, `liq-data` handles data acquisition from external sources. It fetches raw market data from various providers, normalizes it to the shared `liq-core`, and persists it via `liq-store`.

## Architecture: liq-store Integration

**All data access in liq-data goes through liq-store exclusively.** This is a critical architectural requirement that ensures:

- **Single Source of Truth**: All data reads/writes use `ParquetStore` from liq-store
- **Automatic Deduplication**: Append operations merge data by timestamp automatically
- **Consistent Storage Keys**: Bars use `provider/<symbol>/bars/<timeframe>` (e.g., `oanda/EUR_USD/bars/1m`). Quotes/fundamentals/corp actions use `provider/<symbol>/quotes|fundamentals|corp_actions`. Use `liq.store.key_builder` helpers to avoid drift.
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
```

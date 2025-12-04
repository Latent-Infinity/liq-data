# liq-data
Part of the Latent Infinity Quant (LIQ) ecosystem, `liq-data` handles data acquisition from external sources. It fetches raw market data from various providers, normalizes it to the shared `liq-types`, and persists it via `liq-store`.

## QA CLI

Run bar-level QA checks on JSON or Parquet bars:
```bash
python -m liq.data.cli_qa path/to/bars.parquet
```

Outputs:
- Missing ratio
- Zero-volume ratio
- OHLC inconsistencies
- Extreme moves
- Negative volume
- Non-monotonic timestamps

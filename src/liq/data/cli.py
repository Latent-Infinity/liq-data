"""CLI for liq-data using Typer and Rich.

This module is kept for backwards compatibility. The CLI has been
refactored into the liq.data.cli subpackage.

Usage:
    liq-data fetch oanda EUR_USD --start 2005-01-01 --timeframe 1m
    liq-data fetch binance BTC_USDT --start 2020-01-01 --timeframe 1h
    liq-data list oanda
    liq-data info                          # List all available data
    liq-data info oanda EUR_USD            # Show detailed info for symbol
    liq-data validate oanda EUR_USD        # Validate data integrity
    liq-data stats oanda EUR_USD           # Show detailed statistics
    liq-data config                        # Show configuration
"""

# Re-export from the new location for backwards compatibility
from liq.data.cli import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":  # pragma: no cover - exercised via Typer entrypoint
    main()

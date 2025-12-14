"""CLI package for liq-data.

This package provides the command-line interface for fetching and inspecting
market data from various providers.

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

from liq.data.cli.main import app, main

__all__ = ["app", "main"]

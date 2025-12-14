"""
liq-data: Data providers and ingestion for the LIQ Stack.

This package provides market data providers and ingestion pipelines
for fetching and storing financial time-series data.
"""

from liq.data.async_fetcher import AsyncDataFetcher, AsyncRetryPolicy
from liq.data.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DataError,
    ProviderError,
    RateLimitError,
    ValidationError,
)
from liq.data.fetcher import DataFetcher
from liq.data.instruments import InstrumentSync
from liq.data.protocols import AsyncDataProvider, DataProvider
from liq.data.providers import BaseProvider, BinanceProvider, OandaProvider
from liq.data.retry import async_retry, retry
from liq.data.service import DataService
from liq.data.settings import (
    LiqDataSettings,
    create_binance_provider,
    create_oanda_provider,
    get_settings,
)
from liq.data.updater import IncrementalUpdater

__all__ = [
    # Settings
    "LiqDataSettings",
    "get_settings",
    "create_oanda_provider",
    "create_binance_provider",
    # Protocols
    "DataProvider",
    "AsyncDataProvider",
    # Base classes
    "BaseProvider",
    # Providers
    "OandaProvider",
    "BinanceProvider",
    # Service API
    "DataService",
    # Ingestion utilities
    "DataFetcher",
    "AsyncDataFetcher",
    "AsyncRetryPolicy",
    "IncrementalUpdater",
    "InstrumentSync",
    "retry",
    "async_retry",
    # Exceptions
    "DataError",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ValidationError",
    "ConfigurationError",
]

__version__ = "0.1.0"

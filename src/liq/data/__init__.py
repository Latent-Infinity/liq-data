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
    LockboxViolationError,
    ProviderError,
    RateLimitError,
    ValidationError,
    ValidationReuseError,
)
from liq.data.fetcher import DataFetcher
from liq.data.forex import detect_gap_policy, normalize_hourly
from liq.data.fx_session import (
    asian_range_window_utc,
    fx_session_date,
    is_fx_trading_week,
    london_open_utc,
    london_open_window_utc,
    tag_fx_sessions,
)
from liq.data.instruments import InstrumentSync
from liq.data.lockbox import (
    INTRADAY_CAMPAIGN_LEDGER_V1,
    FoldWindows,
    LockboxGuard,
    LockboxLedger,
    resolve_dataset,
)
from liq.data.protocols import AsyncMarketDataProvider, MarketDataProvider
from liq.data.providers import (
    BaseProvider,
    BinanceProvider,
    DatabentoProvider,
    OandaProvider,
)
from liq.data.retry import async_retry, retry
from liq.data.service import DataService
from liq.data.settings import (
    LiqDataSettings,
    create_binance_provider,
    create_databento_provider,
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
    "create_databento_provider",
    # Protocols
    "MarketDataProvider",
    "AsyncMarketDataProvider",
    # Base classes
    "BaseProvider",
    # Providers
    "OandaProvider",
    "BinanceProvider",
    "DatabentoProvider",
    # Service API
    "DataService",
    # Ingestion utilities
    "DataFetcher",
    "AsyncDataFetcher",
    "AsyncRetryPolicy",
    "IncrementalUpdater",
    "InstrumentSync",
    "normalize_hourly",
    "detect_gap_policy",
    "retry",
    "async_retry",
    # FX session tagging
    "asian_range_window_utc",
    "fx_session_date",
    "is_fx_trading_week",
    "london_open_utc",
    "london_open_window_utc",
    "tag_fx_sessions",
    # Lockbox guard
    "LockboxGuard",
    "LockboxLedger",
    "FoldWindows",
    "INTRADAY_CAMPAIGN_LEDGER_V1",
    "resolve_dataset",
    # Exceptions
    "DataError",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ValidationError",
    "ConfigurationError",
    "LockboxViolationError",
    "ValidationReuseError",
]

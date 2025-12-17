"""Settings management for liq-data providers.

This module provides centralized configuration management using pydantic-settings,
automatically loading values from environment variables and .env files.

Design Principles:
    - SRP: Only handles settings/configuration
    - DIP: Providers depend on settings abstraction, not env vars directly
    - KISS: Simple pydantic models with sensible defaults
    - DRY: All data access goes through liq-store (single storage abstraction)

Example:
    from liq.data.settings import get_settings, create_oanda_provider, get_store

    # Get settings (auto-loads from .env)
    settings = get_settings()

    # Create provider from settings
    provider = create_oanda_provider()

    # Get the ParquetStore instance for data access
    store = get_store()
    df = store.read("oanda/EUR_USD/bars/1m")

    # Or access settings directly
    print(settings.oanda_api_key)
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from liq.store import key_builder
from liq.store.parquet import ParquetStore
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_data_root() -> Path:
    """Find the data root directory by searching common locations.

    Search order:
    1. DATA_ROOT environment variable (if set)
    2. Search upward from cwd for 'liq-data/data/financial_data'
    3. Search upward from this file's location for 'data/financial_data'
    4. Fall back to './data/financial_data'

    Returns:
        Path to data root directory
    """
    import os

    # 1. Check environment variable first
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        return Path(env_root)

    # 2. Search upward from cwd for liq-data directory structure
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        # Check for monorepo structure: parent/liq-data/data/financial_data
        candidate = parent / "liq-data" / "data" / "financial_data"
        if candidate.exists():
            return candidate
        # Also check if we're inside liq-data already
        candidate = parent / "data" / "financial_data"
        if candidate.exists() and (parent / "src" / "liq" / "data").exists():
            return candidate

    # 3. Search from this file's location (handles installed package case)
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidate = parent / "data" / "financial_data"
        if candidate.exists():
            return candidate
        # Check sibling liq-data directory
        candidate = parent / "liq-data" / "data" / "financial_data"
        if candidate.exists():
            return candidate

    # 4. Fall back to relative path (will be resolved to absolute later)
    return Path("./data/financial_data")

if TYPE_CHECKING:
    from liq.data.providers.alpaca import AlpacaProvider
    from liq.data.providers.binance import BinanceProvider
    from liq.data.providers.coinbase import CoinbaseProvider
    from liq.data.providers.oanda import OandaProvider
    from liq.data.providers.polygon import PolygonProvider
    from liq.data.providers.tradestation import TradeStationProvider


class LiqDataSettings(BaseSettings):
    """Settings for liq-data providers and storage.

    Settings are loaded from environment variables and .env files.
    Environment variable names match the field names in uppercase
    (e.g., oanda_api_key -> OANDA_API_KEY).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars
    )

    # OANDA settings
    oanda_api_key: str | None = Field(default=None, description="OANDA API key")
    oanda_account_id: str | None = Field(default=None, description="OANDA account ID")
    oanda_environment: str = Field(
        default="practice", description="OANDA environment: 'practice' or 'live'"
    )

    # Binance settings
    binance_api_key: str | None = Field(
        default=None, description="Binance API key (optional for public data)"
    )
    binance_api_secret: str | None = Field(
        default=None, description="Binance API secret (optional for public data)"
    )
    binance_use_us: bool = Field(
        default=False, description="Use Binance.US instead of Binance.com"
    )

    # TradeStation settings (OAuth2)
    tradestation_client_id: str | None = Field(
        default=None, description="TradeStation OAuth2 client ID"
    )
    tradestation_client_secret: str | None = Field(
        default=None, description="TradeStation OAuth2 client secret"
    )
    tradestation_refresh_token: str | None = Field(
        default=None, description="TradeStation OAuth2 refresh token"
    )

    # Coinbase settings
    coinbase_api_key: str | None = Field(
        default=None, description="Coinbase Exchange API key"
    )
    coinbase_api_secret: str | None = Field(
        default=None, description="Coinbase Exchange API secret (base64 encoded)"
    )
    coinbase_passphrase: str | None = Field(
        default=None, description="Coinbase Exchange API passphrase"
    )

    # Polygon settings
    polygon_api_key: str | None = Field(
        default=None, description="Polygon.io API key"
    )

    # Alpaca settings
    alpaca_api_key: str | None = Field(
        default=None, description="Alpaca API key"
    )
    alpaca_api_secret: str | None = Field(
        default=None, description="Alpaca API secret"
    )

    # Storage settings
    data_root: Path = Field(
        default_factory=_find_data_root, description="Root directory for data storage"
    )

    @field_validator("data_root", mode="after")
    @classmethod
    def resolve_data_root(cls, v: Path) -> Path:
        """Resolve data_root to absolute path."""
        return v.resolve()

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="json", description="Log format: 'json' or 'console' for human-readable"
    )
    log_file: Path | None = Field(
        default=None, description="Log file path (optional, logs to console if not set)"
    )


@lru_cache
def get_settings() -> LiqDataSettings:
    """Get cached settings instance.

    Settings are loaded once and cached for subsequent calls.
    Use get_settings.cache_clear() to reload.

    Returns:
        LiqDataSettings instance with values from env/.env file
    """
    return LiqDataSettings()


def create_oanda_provider(settings: LiqDataSettings | None = None) -> "OandaProvider":
    """Create an OANDA provider from settings.

    Args:
        settings: Optional settings instance (uses get_settings() if not provided)

    Returns:
        Configured OandaProvider instance

    Raises:
        ValueError: If required OANDA credentials are not configured
    """
    from liq.data.providers.oanda import OandaProvider

    if settings is None:
        settings = get_settings()

    if not settings.oanda_api_key:
        raise ValueError(
            "OANDA_API_KEY not configured. "
            "Set it in .env file or as environment variable."
        )

    if not settings.oanda_account_id:
        raise ValueError(
            "OANDA_ACCOUNT_ID not configured. "
            "Set it in .env file or as environment variable."
        )

    return OandaProvider(
        api_key=settings.oanda_api_key,
        account_id=settings.oanda_account_id,
        environment=settings.oanda_environment,
    )


def create_binance_provider(settings: LiqDataSettings | None = None) -> "BinanceProvider":
    """Create a Binance provider from settings.

    Args:
        settings: Optional settings instance (uses get_settings() if not provided)

    Returns:
        Configured BinanceProvider instance

    Note:
        Binance public market data API doesn't require authentication.
        API keys are only needed for trading/account endpoints.
    """
    from liq.data.providers.binance import BinanceProvider

    if settings is None:
        settings = get_settings()

    return BinanceProvider(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
        use_us=settings.binance_use_us,
    )


def create_tradestation_provider(
    settings: LiqDataSettings | None = None,
) -> "TradeStationProvider":
    """Create a TradeStation provider from settings.

    Args:
        settings: Optional settings instance (uses get_settings() if not provided)

    Returns:
        Configured TradeStationProvider instance

    Raises:
        ValueError: If required TradeStation credentials are not configured
    """
    from liq.data.providers.tradestation import TradeStationProvider

    if settings is None:
        settings = get_settings()

    if not settings.tradestation_client_id:
        raise ValueError(
            "TRADESTATION_CLIENT_ID not configured. "
            "Set it in .env file or as environment variable."
        )

    if not settings.tradestation_client_secret:
        raise ValueError(
            "TRADESTATION_CLIENT_SECRET not configured. "
            "Set it in .env file or as environment variable."
        )

    if not settings.tradestation_refresh_token:
        raise ValueError(
            "TRADESTATION_REFRESH_TOKEN not configured. "
            "Set it in .env file or as environment variable."
        )

    return TradeStationProvider(
        client_id=settings.tradestation_client_id,
        client_secret=settings.tradestation_client_secret,
        refresh_token=settings.tradestation_refresh_token,
    )


def create_coinbase_provider(
    settings: LiqDataSettings | None = None,
) -> "CoinbaseProvider":
    """Create a Coinbase provider from settings.

    Args:
        settings: Optional settings instance (uses get_settings() if not provided)

    Returns:
        Configured CoinbaseProvider instance

    Raises:
        ValueError: If required Coinbase credentials are not configured
    """
    from liq.data.providers.coinbase import CoinbaseProvider

    if settings is None:
        settings = get_settings()

    if not settings.coinbase_api_key:
        raise ValueError(
            "COINBASE_API_KEY not configured. "
            "Set it in .env file or as environment variable."
        )

    if not settings.coinbase_api_secret:
        raise ValueError(
            "COINBASE_API_SECRET not configured. "
            "Set it in .env file or as environment variable."
        )

    if not settings.coinbase_passphrase:
        raise ValueError(
            "COINBASE_PASSPHRASE not configured. "
            "Set it in .env file or as environment variable."
        )

    return CoinbaseProvider(
        api_key=settings.coinbase_api_key,
        api_secret=settings.coinbase_api_secret,
        passphrase=settings.coinbase_passphrase,
    )


def create_polygon_provider(
    settings: LiqDataSettings | None = None,
) -> "PolygonProvider":
    """Create a Polygon provider from settings.

    Args:
        settings: Optional settings instance (uses get_settings() if not provided)

    Returns:
        Configured PolygonProvider instance

    Raises:
        ValueError: If required Polygon credentials are not configured
    """
    from liq.data.providers.polygon import PolygonProvider

    if settings is None:
        settings = get_settings()

    if not settings.polygon_api_key:
        raise ValueError(
            "POLYGON_API_KEY not configured. "
            "Set it in .env file or as environment variable."
        )

    return PolygonProvider(
        api_key=settings.polygon_api_key,
    )


def create_alpaca_provider(
    settings: LiqDataSettings | None = None,
) -> "AlpacaProvider":
    """Create an Alpaca provider from settings.

    Args:
        settings: Optional settings instance (uses get_settings() if not provided)

    Returns:
        Configured AlpacaProvider instance

    Raises:
        ValueError: If required Alpaca credentials are not configured
    """
    from liq.data.providers.alpaca import AlpacaProvider

    if settings is None:
        settings = get_settings()

    if not settings.alpaca_api_key:
        raise ValueError(
            "ALPACA_API_KEY not configured. "
            "Set it in .env file or as environment variable."
        )

    if not settings.alpaca_api_secret:
        raise ValueError(
            "ALPACA_API_SECRET not configured. "
            "Set it in .env file or as environment variable."
        )

    return AlpacaProvider(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
    )


@lru_cache
def get_store() -> ParquetStore:
    """Get the ParquetStore instance for data access.

    This is the canonical way to access data storage in liq-data.
    All data operations should go through this store.

    Returns:
        ParquetStore instance configured with data_root

    Example:
        >>> store = get_store()
        >>> df = store.read("oanda/EUR_USD/1m")
    """
    settings = get_settings()
    return ParquetStore(str(settings.data_root))


def get_storage_key(provider: str, symbol: str, timeframe: str) -> str:
    """Get the storage key for a data series.

    Args:
        provider: Provider name (e.g., "oanda", "binance")
        symbol: Symbol name (e.g., "EUR_USD", "BTC_USDT")
        timeframe: Timeframe (e.g., "1m", "1h", "1d")

    Returns:
        Storage key string (e.g., "oanda/EUR_USD/bars/1m")

    Example:
        >>> key = get_storage_key("oanda", "EUR_USD", "1m")
        >>> key
        'oanda/EUR_USD/bars/1m'
    """
    return f"{provider}/{key_builder.bars(symbol, timeframe)}"


def load_symbol_data(
    provider: str,
    symbol: str,
    timeframe: str,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Load data for a symbol from the data store.

    This is the primary interface for accessing market data by symbol.
    Uses liq-store for all data access operations.

    Args:
        provider: Provider name (e.g., "oanda", "binance")
        symbol: Symbol name (e.g., "EUR_USD", "BTC_USDT")
        timeframe: Timeframe (e.g., "1m", "1h", "1d")
        start: Optional start date filter (inclusive)
        end: Optional end date filter (inclusive)

    Returns:
        DataFrame with OHLCV data

    Raises:
        FileNotFoundError: If data doesn't exist

    Example:
        >>> df = load_symbol_data("oanda", "EUR_USD", "1m")
        >>> len(df)
        7690495
        >>> df = load_symbol_data("binance", "BTC_USDT", "1m", start=date(2024, 1, 1))
        >>> df["timestamp"].min()
        datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    store = get_store()
    storage_key = get_storage_key(provider, symbol, timeframe)

    if not store.exists(storage_key):
        raise FileNotFoundError(
            f"Data not found for {provider}/{symbol}/{timeframe}. "
            f"Run: liq-data fetch {provider} {symbol} --timeframe {timeframe}"
        )
    return store.read(storage_key, start=start, end=end)


def list_available_data() -> list[dict[str, str]]:
    """List all available data in the data store.

    Uses liq-store for data discovery.

    Returns:
        List of dicts with provider, symbol, timeframe info

    Example:
        >>> data = list_available_data()
        >>> data[0]
        {'provider': 'oanda', 'symbol': 'EUR_USD', 'timeframe': '1m'}
    """
    store = get_store()
    keys = store.list_keys()

    result = []
    for key in keys:
        parts = key.split("/")
        if len(parts) >= 4 and parts[2] == "bars":
            result.append({
                "provider": parts[0],
                "symbol": parts[1],
                "timeframe": parts[3],
            })
        elif len(parts) >= 3:
            result.append({
                "provider": parts[0],
                "symbol": parts[1],
                "timeframe": parts[2],
            })

    return result

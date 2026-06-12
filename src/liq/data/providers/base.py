"""Base provider implementation for the LIQ Stack.

This module provides a base class for data providers that implements
common functionality while allowing subclasses to override specific behavior.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import polars as pl

from liq.data.exceptions import ProviderError

# Standard Polars types for OHLCV data
# Using Decimal for financial precision:
# - PRICE: 38 digits precision, 8 decimal places (handles crypto with many decimals)
# - VOLUME: 38 digits precision, 2 decimal places (sub-unit volumes)
PRICE_DTYPE = pl.Decimal(precision=38, scale=8)
VOLUME_DTYPE = pl.Decimal(precision=38, scale=2)


class BaseProvider(ABC):
    """Abstract base class for data providers.

    Provides common functionality for all data providers including:
    - Standard timeframe definitions
    - DataFrame conversion utilities
    - Error handling patterns

    Subclasses must implement the abstract methods to provide
    provider-specific API interactions.

    Example:
        class MyProvider(BaseProvider):
            @property
            def name(self) -> str:
                return "my_provider"

            def _fetch_bars_impl(self, symbol, start, end, timeframe):
                # Provider-specific implementation
                ...
    """

    # Standard timeframes supported by most providers
    STANDARD_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    def supported_timeframes(self) -> list[str]:
        """Return supported timeframes for this provider.

        Override in subclasses if different timeframes are supported.
        """
        return self.STANDARD_TIMEFRAMES

    @property
    def supported_asset_classes(self) -> list[str]:
        """Asset classes supported by this provider."""
        return ["forex"]

    def validate_credentials(self) -> bool:
        """Validate configured credentials (default: assume valid)."""
        return True

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch OHLCV bar data for a symbol.

        Args:
            symbol: Instrument symbol (provider-specific format)
            start: Start date (inclusive)
            end: End date (inclusive)
            timeframe: Bar timeframe (e.g., "1m", "1h", "1d")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ProviderError: If fetch fails
        """
        ...

    @abstractmethod
    def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        """List available instruments from the provider.

        Args:
            asset_class: Optional filter by asset class

        Returns:
            DataFrame with instrument metadata
        """
        ...

    def get_instrument(self, symbol: str) -> dict[str, Any] | None:
        """Get metadata for a specific instrument.

        Default implementation filters list_instruments().
        Override for more efficient provider-specific lookup.

        Args:
            symbol: Instrument symbol

        Returns:
            Dict with instrument metadata or None if not found
        """
        instruments = self.list_instruments()
        if instruments.is_empty():
            return None

        if "symbol" not in instruments.columns:
            return None

        filtered = instruments.filter(pl.col("symbol") == symbol)
        if filtered.is_empty():
            return None

        return filtered.row(0, named=True)

    # ------------------------------------------------------------------
    # Optional-API default implementations.
    #
    # ``MarketDataProvider`` (protocols.py) declares the full surface a
    # ``DataService`` may consume. Concrete adapters override only the
    # methods their venue actually supports. The defaults below raise
    # ``NotImplementedError`` at call time so structural typing passes
    # without quietly returning misleading data — callers must catch or
    # handle the exception explicitly.
    # ------------------------------------------------------------------

    def fetch_quotes(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Default: not implemented. Override in adapters that support quotes."""
        raise NotImplementedError(f"{type(self).__name__}.fetch_quotes is not implemented")

    def fetch_fundamentals(self, symbol: str, as_of: date) -> dict[str, Any]:
        """Default: not implemented. Override in adapters that support fundamentals."""
        raise NotImplementedError(f"{type(self).__name__}.fetch_fundamentals is not implemented")

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[Any]:
        """Default: not implemented. Override in adapters that support CA."""
        raise NotImplementedError(f"{type(self).__name__}.get_corporate_actions is not implemented")

    def get_universe(self, asset_class: str, as_of: date | None = None) -> list[str]:
        """Default: not implemented. Override in adapters that expose a universe."""
        raise NotImplementedError(f"{type(self).__name__}.get_universe is not implemented")

    def fetch_instruments(self, asset_class: str) -> list[Any]:
        """Default: not implemented. Override in adapters that expose instruments."""
        raise NotImplementedError(f"{type(self).__name__}.fetch_instruments is not implemented")

    def validate_timeframe(self, timeframe: str) -> None:
        """Validate that a timeframe is supported.

        Args:
            timeframe: Timeframe to validate

        Raises:
            ProviderError: If timeframe is not supported
        """
        if timeframe not in self.supported_timeframes:
            raise ProviderError(
                f"Unsupported timeframe: {timeframe}. Supported: {self.supported_timeframes}"
            )

    @staticmethod
    def bars_to_dataframe(bars: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert a list of bar dictionaries to a Polars DataFrame.

        Args:
            bars: List of dicts with keys: timestamp, open, high, low, close, volume

        Returns:
            Polars DataFrame with standardized schema using Decimal for precision
        """
        if not bars:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime("us", "UTC"),
                    "open": PRICE_DTYPE,
                    "high": PRICE_DTYPE,
                    "low": PRICE_DTYPE,
                    "close": PRICE_DTYPE,
                    "volume": VOLUME_DTYPE,
                }
            )

        return pl.DataFrame(bars).select(
            [
                pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
                pl.col("open").cast(PRICE_DTYPE),
                pl.col("high").cast(PRICE_DTYPE),
                pl.col("low").cast(PRICE_DTYPE),
                pl.col("close").cast(PRICE_DTYPE),
                pl.col("volume").cast(VOLUME_DTYPE),
            ]
        )

"""Protocols for data providers in the LIQ Stack.

This module defines the DataProvider protocol that all data providers
must implement to be usable with the LIQ data pipeline.
"""

from datetime import date
from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for market data providers.

    All data providers must implement this protocol to be compatible
    with the LIQ data pipeline.

    Example:
        class MyProvider:
            @property
            def name(self) -> str:
                return "my_provider"

            def fetch_bars(
                self,
                symbol: str,
                start: date,
                end: date,
                timeframe: str = "1d",
            ) -> pl.DataFrame:
                # Fetch data from provider API
                ...
    """

    @property
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    def supported_asset_classes(self) -> list[str]:
        """Asset classes supported by the provider (e.g., ['forex'])."""
        ...

    def validate_credentials(self) -> bool:
        """Validate configured credentials; return True if usable."""
        ...

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
            Empty DataFrame if no data available

        Raises:
            ProviderError: If fetch fails
            RateLimitError: If rate limit exceeded
        """
        ...

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

        Args:
            symbol: Instrument symbol

        Returns:
            Dict with instrument metadata or None if not found
        """
        ...

    def fetch_quotes(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Fetch quote data when supported."""
        ...

    def fetch_fundamentals(self, symbol: str, as_of: date) -> dict[str, Any]:
        """Fetch point-in-time fundamentals if supported."""
        ...

    def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[Any]:
        """Fetch corporate actions (splits/dividends) if supported."""
        ...

    def get_universe(self, asset_class: str, as_of: date | None = None) -> list[str]:
        """List tradeable symbols (survivorship-bias free) for an asset class."""
        ...

    def fetch_instruments(self, asset_class: str) -> list[Any]:
        """Fetch instruments metadata for an asset class."""
        ...


@runtime_checkable
class AsyncDataProvider(Protocol):
    """Protocol for async market data providers.

    Same as DataProvider but with async methods for concurrent fetching.
    """

    @property
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    def supported_asset_classes(self) -> list[str]:
        """Asset classes supported by the provider (e.g., ['forex'])."""
        ...

    async def validate_credentials(self) -> bool:
        """Validate configured credentials; return True if usable."""
        ...

    async def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> pl.DataFrame:
        """Fetch OHLCV bar data for a symbol asynchronously."""
        ...

    async def list_instruments(self, asset_class: str | None = None) -> pl.DataFrame:
        """List available instruments from the provider asynchronously."""
        ...

    async def get_instrument(self, symbol: str) -> dict[str, Any] | None:
        """Get metadata for a specific instrument asynchronously."""
        ...

    async def fetch_quotes(self, symbol: str, start: date, end: date) -> pl.DataFrame:
        """Fetch quote data asynchronously when supported."""
        ...

    async def fetch_fundamentals(self, symbol: str, as_of: date) -> dict[str, Any]:
        """Fetch fundamentals asynchronously if supported."""
        ...

    async def get_corporate_actions(self, symbol: str, start: date, end: date) -> list[Any]:
        """Fetch corporate actions asynchronously if supported."""
        ...

    async def get_universe(self, asset_class: str, as_of: date | None = None) -> list[str]:
        """List tradeable symbols asynchronously for an asset class."""
        ...

    async def fetch_instruments(self, asset_class: str) -> list[Any]:
        """Fetch instruments metadata asynchronously."""
        ...

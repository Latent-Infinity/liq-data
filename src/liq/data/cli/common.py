"""Common utilities for CLI commands.

This module provides shared functionality used across CLI commands including
date parsing, provider factory, and Rich console setup.
"""

from datetime import date, datetime

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from liq.data.protocols import DataProvider
from liq.data.settings import (
    create_alpaca_provider,
    create_binance_provider,
    create_coinbase_provider,
    create_oanda_provider,
    create_polygon_provider,
    create_tradestation_provider,
    get_settings,
)
from liq.store import key_builder

# Shared console instance
console = Console()


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def storage_key(provider: str, symbol: str, timeframe: str) -> str:
    """Provider-prefixed bars key via key_builder."""
    return f"{provider}/{key_builder.bars(symbol, timeframe)}"


def get_provider(provider_name: str) -> DataProvider:
    """Create a data provider instance by name.

    Args:
        provider_name: Provider name (oanda, binance, tradestation, coinbase, polygon, alpaca)

    Returns:
        Configured DataProvider instance

    Raises:
        typer.Exit: If provider is unknown or configuration is invalid
    """
    try:
        provider_lower = provider_name.lower()
        if provider_lower == "oanda":
            return create_oanda_provider()
        elif provider_lower == "binance":
            return create_binance_provider()
        elif provider_lower == "tradestation":
            return create_tradestation_provider()
        elif provider_lower == "coinbase":
            return create_coinbase_provider()
        elif provider_lower == "polygon":
            return create_polygon_provider()
        elif provider_lower == "alpaca":
            return create_alpaca_provider()
        else:
            console.print(f"[red]Unknown provider: {provider_name}[/red]")
            raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)


def create_fetch_progress() -> Progress:
    """Create a Rich progress bar for data fetching."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


# Standard timeframe to minutes mapping
TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


def parse_source_spec(source: str) -> tuple[str, str]:
    """Parse a source specification in 'provider:symbol' format.

    Args:
        source: Source specification (e.g., 'oanda:EUR_USD')

    Returns:
        Tuple of (provider, symbol)

    Raises:
        ValueError: If format is invalid
    """
    if ":" not in source:
        raise ValueError(
            f"Invalid source format: '{source}'. "
            "Expected 'provider:symbol' (e.g., 'oanda:EUR_USD')"
        )
    parts = source.split(":", 1)
    return parts[0].lower(), parts[1]

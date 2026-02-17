"""Info commands for liq-data CLI.

Commands for listing and inspecting market data.

All data access uses liq-store for consistent storage abstraction.
"""

from typing import TYPE_CHECKING, Annotated

import polars as pl
import typer
from rich.table import Table

from liq.data.cli.common import console, get_provider
from liq.data.settings import get_settings, get_storage_key, get_store

if TYPE_CHECKING:
    from liq.store.parquet import ParquetStore

app = typer.Typer()


@app.command("list")
def list_instruments(
    provider: Annotated[str, typer.Argument(help="Provider: 'oanda', 'binance', 'tradestation', 'coinbase', 'polygon', or 'alpaca'")],
    asset_class: Annotated[
        str | None, typer.Option("--asset-class", "-a", help="Filter by asset class")
    ] = None,
) -> None:
    """List available instruments from a provider."""
    console.print(f"\n[bold blue]Listing instruments from {provider}[/bold blue]\n")

    data_provider = get_provider(provider)

    with console.status("Fetching instruments..."):
        try:
            df = data_provider.list_instruments(asset_class)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Display as table
    table = Table(title=f"{provider.upper()} Instruments ({len(df)} total)")
    table.add_column("Symbol", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Asset Class", style="yellow")
    table.add_column("Type", style="magenta")

    for row in df.iter_rows(named=True):
        table.add_row(
            row.get("symbol", ""),
            row.get("name", ""),
            row.get("asset_class", ""),
            row.get("type", ""),
        )

    console.print(table)


@app.command("config")
def show_config() -> None:
    """Show current configuration."""
    settings = get_settings()

    def _mask(value: object, missing_label: str) -> str:
        if isinstance(value, str):
            return "***" if value else missing_label
        return "***" if value else missing_label

    def _render(value: object, missing_label: str) -> str:
        if value is None:
            return missing_label
        if isinstance(value, str):
            return value if value else missing_label
        return str(value)

    table = Table(title="LIQ Data Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("OANDA API Key", _mask(settings.oanda_api_key, "[red]Not set[/red]"))
    table.add_row("OANDA Account ID", _render(settings.oanda_account_id, "[red]Not set[/red]"))
    table.add_row("OANDA Environment", _render(settings.oanda_environment, "[red]Not set[/red]"))
    table.add_row("Binance API Key", _mask(settings.binance_api_key, "[dim]Not set[/dim]"))
    table.add_row("Binance Use US", _render(settings.binance_use_us, "[dim]Not set[/dim]"))
    table.add_row("TradeStation Client ID", _mask(settings.tradestation_client_id, "[dim]Not set[/dim]"))
    table.add_row("TradeStation Refresh Token", _mask(settings.tradestation_refresh_token, "[dim]Not set[/dim]"))
    table.add_row("TradeStation Redirect URI", _render(settings.tradestation_redirect_uri, "[dim]Not set[/dim]"))
    table.add_row("TradeStation Scopes", _render(settings.tradestation_scopes, "[dim]Not set[/dim]"))
    table.add_row(
        "TradeStation Persist Refresh Token",
        _render(settings.tradestation_persist_refresh_token, "[dim]Not set[/dim]"),
    )
    table.add_row("Coinbase API Key", _mask(settings.coinbase_api_key, "[dim]Not set[/dim]"))
    table.add_row("Coinbase Passphrase", _mask(settings.coinbase_passphrase, "[dim]Not set[/dim]"))
    table.add_row("Polygon API Key", _mask(settings.polygon_api_key, "[dim]Not set[/dim]"))
    table.add_row("Data Root", _render(settings.data_root, "[dim]Not set[/dim]"))
    table.add_row("Log Level", _render(settings.log_level, "[dim]Not set[/dim]"))
    table.add_row("Log Format", _render(settings.log_format, "[dim]Not set[/dim]"))
    table.add_row("Log File", _render(settings.log_file, "[dim]None[/dim]"))

    console.print(table)


@app.command("info")
def show_data_info(
    provider: Annotated[
        str | None, typer.Argument(help="Provider (e.g., 'oanda', 'binance')")
    ] = None,
    symbol: Annotated[
        str | None, typer.Argument(help="Symbol (e.g., 'EUR_USD')")
    ] = None,
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
) -> None:
    """Show information about available data.

    Without arguments, lists all available data.
    With provider/symbol/timeframe, shows detailed info for that data.
    """
    get_settings()

    # If specific symbol requested, show detailed info
    if provider and symbol:
        store = get_store()
        storage_key = get_storage_key(provider, symbol, timeframe)
        if not store.exists(storage_key):
            console.print(f"[red]Data not found: {storage_key}[/red]")
            console.print(f"[yellow]Run: liq-data fetch {provider} {symbol} --timeframe {timeframe}[/yellow]")
            raise typer.Exit(1)
        _show_symbol_info(store, storage_key, provider, symbol, timeframe)
        return

    # Otherwise, list all available data using liq-store
    store = get_store()
    keys = store.list_keys()

    if not keys:
        console.print("[yellow]No data available. Use 'liq-data fetch' to download data.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Available Data ({len(keys)} datasets)")
    table.add_column("Provider", style="cyan")
    table.add_column("Symbol", style="green")
    table.add_column("Timeframe", style="yellow")
    table.add_column("Bars", style="magenta", justify="right")
    table.add_column("Start", style="blue")
    table.add_column("End", style="blue")

    for key in sorted(keys):
        try:
            parts = key.split("/")
            if len(parts) < 3:
                continue

            prov = parts[0]
            sym = parts[1]
            tf = parts[2]

            # Use store to get date range without loading full data
            date_range = store.get_date_range(key)
            df = store.read(key, columns=["timestamp"])

            table.add_row(
                prov,
                sym,
                tf,
                f"{len(df):,}",
                str(date_range[0]) if date_range else "N/A",
                str(date_range[1]) if date_range else "N/A",
            )
        except Exception as e:
            table.add_row(key, "Error", str(e), "", "", "")

    console.print(table)


def _show_symbol_info(
    store: "ParquetStore",
    storage_key: str,
    provider: str,
    symbol: str,
    timeframe: str,
) -> None:
    """Show detailed info for a symbol using liq-store."""
    df = store.read(storage_key)

    console.print(f"\n[bold blue]{provider}/{symbol}/{timeframe}[/bold blue]\n")

    # Basic stats
    table = Table(title="Data Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Bars", f"{len(df):,}")
    table.add_row("Columns", ", ".join(df.columns))

    date_range = store.get_date_range(storage_key)
    if date_range:
        table.add_row("First Timestamp", str(date_range[0]))
        table.add_row("Last Timestamp", str(date_range[1]))

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min is not None and col_max is not None:
                table.add_row(f"{col.title()} Range", f"{col_min!s} - {col_max!s}")

    if "volume" in df.columns:
        vol_sum = df["volume"].sum()
        if vol_sum is not None:
            table.add_row("Total Volume", f"{vol_sum:,.0f}")

    console.print(table)


@app.command("stats")
def show_stats(
    provider: Annotated[str, typer.Argument(help="Provider (e.g., 'oanda', 'binance')")],
    symbol: Annotated[str, typer.Argument(help="Symbol (e.g., 'EUR_USD')")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
) -> None:
    """Show detailed statistics for a symbol's data."""
    store = get_store()
    storage_key = get_storage_key(provider, symbol, timeframe)
    if not store.exists(storage_key):
        console.print(f"[red]Data not found: {provider}/{symbol}/{timeframe}[/red]")
        console.print(f"[yellow]Run: liq-data fetch {provider} {symbol} --timeframe {timeframe}[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Statistics: {provider}/{symbol}/{timeframe}[/bold blue]\n")

    df = store.read(storage_key)

    # OHLCV statistics
    stats_table = Table(title="OHLCV Statistics")
    stats_table.add_column("Column", style="cyan")
    stats_table.add_column("Min", justify="right")
    stats_table.add_column("Max", justify="right")
    stats_table.add_column("Mean", justify="right")
    stats_table.add_column("Std", justify="right")
    stats_table.add_column("Nulls", justify="right")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()
            col_min = float(min_val) if min_val is not None else 0.0  # type: ignore[arg-type]
            col_max = float(max_val) if max_val is not None else 0.0  # type: ignore[arg-type]
            col_mean = float(mean_val) if mean_val is not None else 0.0  # type: ignore[arg-type]
            col_std = float(std_val) if std_val is not None else 0.0  # type: ignore[arg-type]
            stats_table.add_row(
                col,
                f"{col_min:.5f}" if col != "volume" else f"{col_min:,.0f}",
                f"{col_max:.5f}" if col != "volume" else f"{col_max:,.0f}",
                f"{col_mean:.5f}" if col != "volume" else f"{col_mean:,.0f}",
                f"{col_std:.5f}" if col != "volume" else f"{col_std:,.0f}",
                f"{df[col].null_count():,}",
            )

    console.print(stats_table)

    # Time coverage
    time_table = Table(title="Time Coverage")
    time_table.add_column("Metric", style="cyan")
    time_table.add_column("Value", style="green")

    time_table.add_row("First Bar", str(df["timestamp"].min()))
    time_table.add_row("Last Bar", str(df["timestamp"].max()))

    # Calculate coverage by year
    df_years = df.with_columns([
        pl.col("timestamp").dt.year().alias("year")
    ]).group_by("year").agg(pl.len().alias("count"))

    years_str = ", ".join([
        f"{row['year']}: {row['count']:,}"
        for row in df_years.sort("year").iter_rows(named=True)
    ])
    time_table.add_row("Bars by Year", years_str[:100] + "..." if len(years_str) > 100 else years_str)

    console.print(time_table)

    # Show yearly breakdown
    yearly_table = Table(title="Yearly Breakdown")
    yearly_table.add_column("Year", style="cyan", justify="right")
    yearly_table.add_column("Bars", style="green", justify="right")
    yearly_table.add_column("% of Total", style="yellow", justify="right")

    for row in df_years.sort("year").iter_rows(named=True):
        pct = row["count"] / len(df) * 100
        yearly_table.add_row(str(row["year"]), f"{row['count']:,}", f"{pct:.1f}%")

    console.print(yearly_table)

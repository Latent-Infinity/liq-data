"""Fetch commands for liq-data CLI using DataService."""

from datetime import date
from typing import Annotated

import typer

from liq.data.cli.common import console, parse_date
from liq.data.service import DataService
from liq.data.exceptions import ProviderError, ConfigurationError

app = typer.Typer(help="Commands for fetching and updating market data")


@app.command("fetch")
def fetch_bars(
    provider: Annotated[str, typer.Argument(help="Provider: oanda, binance, tradestation, coinbase, polygon, alpaca")],
    symbol: Annotated[str, typer.Argument(help="Symbol (e.g., EUR_USD, BTC_USDT, AAPL, BTC-USD)")],
    start: Annotated[str, typer.Option("--start", "-s", help="Start date (YYYY-MM-DD)")],
    end: Annotated[str, typer.Option("--end", "-e", help="End date (YYYY-MM-DD)")] = "",
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
    mode: Annotated[str, typer.Option("--mode", "-m", help="Write mode: append|overwrite")] = "append",
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, do not save"),
) -> None:
    """Fetch bars via DataService (liq-store backed)."""
    ds = DataService()
    start_date = parse_date(start)
    end_date = parse_date(end) if end else date.today()

    console.print(f"\n[bold blue]Fetching {symbol} from {provider} [{timeframe}] {start_date} -> {end_date}[/bold blue]")
    try:
        df = ds.fetch(provider, symbol, start_date, end_date, timeframe=timeframe, save=not dry_run, mode=mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except (ProviderError, ConfigurationError) as exc:
        console.print(f"[red]Provider error: {exc}[/red]")
        raise typer.Exit(1)

    if df.is_empty():
        console.print("[yellow]No data returned[/yellow]")
        raise typer.Exit(0)

    console.print(f"[green]Fetched {len(df):,} rows[/green]")
    if dry_run:
        console.print(df.head())
        return

    console.print("[green]Stored via liq-store[/green]")


@app.command("backfill")
def backfill_bars(
    provider: Annotated[str, typer.Argument(help="Provider")],
    symbol: Annotated[str, typer.Argument(help="Symbol")],
    start: Annotated[str, typer.Option("--start", "-s", help="Start date (YYYY-MM-DD)")],
    end: Annotated[str, typer.Option("--end", "-e", help="End date (YYYY-MM-DD)")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
) -> None:
    """Backfill missing bars based on gaps."""
    ds = DataService()
    start_date = parse_date(start)
    end_date = parse_date(end)
    console.print(f"\n[bold blue]Backfilling {symbol} {timeframe} from {start_date} to {end_date}[/bold blue]")
    df = ds.backfill(provider, symbol, start=start_date, end=end_date, timeframe=timeframe)
    console.print(f"[green]Backfill complete. Rows now: {len(df)}[/green]")

"""Fetch commands for liq-data CLI using DataService."""

import json
from datetime import date
from typing import Annotated

import typer

from liq.data.cli.common import console, parse_date
from liq.data.exceptions import ConfigurationError, ProviderError
from liq.data.service import DataService

app = typer.Typer(help="Commands for fetching and updating market data")


@app.command("fetch")
def fetch_bars(
    provider_arg: Annotated[
        str | None,
        typer.Argument(
            help=("Provider: oanda, binance, tradestation, coinbase, polygon, alpaca, databento")
        ),
    ] = None,
    symbol_arg: Annotated[
        str | None,
        typer.Argument(help="Symbol (e.g., EUR_USD, BTC_USDT, AAPL, BTC-USD)"),
    ] = None,
    provider_opt: Annotated[
        str | None,
        typer.Option("--provider", help="Provider name; alternative to positional provider"),
    ] = None,
    symbols_opt: Annotated[
        str | None,
        typer.Option(
            "--symbols", help="Comma-separated symbol list; alternative to positional symbol"
        ),
    ] = None,
    start: Annotated[str, typer.Option("--start", "-s", help="Start date (YYYY-MM-DD)")] = "",
    end: Annotated[str, typer.Option("--end", "-e", help="End date (YYYY-MM-DD)")] = "",
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="Write mode: append|overwrite")
    ] = "append",
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only, do not save"),
    output: Annotated[str, typer.Option("--output", help="Output format: text|json")] = "text",
) -> None:
    """Fetch bars via DataService (liq-store backed)."""
    provider = _resolve_provider(provider_arg=provider_arg, provider_opt=provider_opt)
    symbols = _resolve_symbols(symbol_arg=symbol_arg, symbols_opt=symbols_opt)
    if not start:
        console.print("[red]--start is required (YYYY-MM-DD).[/red]")
        raise typer.Exit(1)
    if output not in {"text", "json"}:
        console.print(f"[red]Unsupported output format: {output}. Expected text or json.[/red]")
        raise typer.Exit(1)

    ds = DataService()
    start_date = parse_date(start)
    end_date = parse_date(end) if end else date.today()

    summaries: list[dict[str, object]] = []
    for symbol in symbols:
        if output == "text":
            console.print(
                f"\n[bold blue]Fetching {symbol} from {provider} [{timeframe}] "
                f"{start_date} -> {end_date}[/bold blue]"
            )
        try:
            df = ds.fetch(
                provider,
                symbol,
                start_date,
                end_date,
                timeframe=timeframe,
                save=not dry_run,
                mode=mode,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        except (ProviderError, ConfigurationError) as exc:
            console.print(f"[red]Provider error: {exc}[/red]")
            raise typer.Exit(1)

        summaries.append(
            {
                "provider": provider,
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "rows": df.height,
                "stored": not dry_run and not df.is_empty(),
            }
        )

        if output == "json":
            continue

        if df.is_empty():
            console.print("[yellow]No data returned[/yellow]")
            continue

        console.print(f"[green]Fetched {len(df):,} rows[/green]")
        if dry_run:
            console.print(df.head())
            continue

        console.print("[green]Stored via liq-store[/green]")

    if output == "json":
        console.print(json.dumps({"fetches": summaries}, sort_keys=True))
        raise typer.Exit(0)


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
    console.print(
        f"\n[bold blue]Backfilling {symbol} {timeframe} from {start_date} to {end_date}[/bold blue]"
    )
    df = ds.backfill(provider, symbol, start=start_date, end=end_date, timeframe=timeframe)
    console.print(f"[green]Backfill complete. Rows now: {len(df)}[/green]")


def _resolve_provider(*, provider_arg: str | None, provider_opt: str | None) -> str:
    if provider_arg and provider_opt and provider_arg != provider_opt:
        console.print(
            f"[red]Provider mismatch: positional {provider_arg!r} != --provider {provider_opt!r}[/red]"
        )
        raise typer.Exit(1)
    provider = provider_opt or provider_arg
    if not provider:
        console.print("[red]Provider is required. Pass it positionally or via --provider.[/red]")
        raise typer.Exit(1)
    return provider


def _resolve_symbols(*, symbol_arg: str | None, symbols_opt: str | None) -> list[str]:
    if symbol_arg and symbols_opt and symbol_arg != symbols_opt:
        console.print(
            f"[red]Symbol mismatch: positional {symbol_arg!r} != --symbols {symbols_opt!r}[/red]"
        )
        raise typer.Exit(1)
    raw = symbols_opt or symbol_arg
    if not raw:
        console.print("[red]Symbol is required. Pass it positionally or via --symbols.[/red]")
        raise typer.Exit(1)
    symbols = [s.strip() for s in raw.split(",") if s.strip()]
    if not symbols:
        console.print("[red]At least one non-empty symbol is required.[/red]")
        raise typer.Exit(1)
    return symbols

"""Data management commands for liq-data CLI.

Commands for comparing and deleting market data.
"""

import json
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.table import Table

from liq.data.cli.common import console, parse_source_spec
from liq.data.settings import get_settings, get_store, get_storage_key
from liq.data.service import DataService

app = typer.Typer()


@app.command("compare")
def compare_data(
    source1: Annotated[str, typer.Argument(help="First source: 'provider:symbol' (e.g., 'oanda:EUR_USD')")],
    source2: Annotated[str, typer.Argument(help="Second source: 'provider:symbol' (e.g., 'polygon:EUR_USD')")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file (CSV or JSON)")] = None,
) -> None:
    """Compare data between two sources (cross-provider or cross-symbol).

    Sources use the format 'provider:symbol' (e.g., 'oanda:EUR_USD').

    Examples:
        liq-data compare oanda:EUR_USD polygon:EUR_USD  # Cross-provider
        liq-data compare oanda:EUR_USD oanda:GBP_USD    # Cross-symbol
        liq-data compare binance:BTC_USDT coinbase:BTC-USD  # Cross-venue
    """
    store = get_store()

    # Parse source specifications
    try:
        prov1, sym1 = parse_source_spec(source1)
        prov2, sym2 = parse_source_spec(source2)
    except ValueError as e:
        console.print(f"[red]Invalid format: {e}[/red]")
        raise typer.Exit(1)

    # Load data via liq-store
    key1 = get_storage_key(prov1, sym1, timeframe)
    key2 = get_storage_key(prov2, sym2, timeframe)

    if not store.exists(key1):
        console.print(f"[red]Data not found: {prov1}/{sym1}/{timeframe}[/red]")
        raise typer.Exit(1)

    if not store.exists(key2):
        console.print(f"[red]Data not found: {prov2}/{sym2}/{timeframe}[/red]")
        raise typer.Exit(1)

    df1 = store.read(key1)
    df2 = store.read(key2)

    console.print(f"\n[bold blue]Comparison: {source1} vs {source2}[/bold blue]")
    console.print(f"  Timeframe: {timeframe}\n")

    # Show source info
    info_table = Table(title="Source Information")
    info_table.add_column("Metric", style="cyan")
    info_table.add_column(source1, style="green")
    info_table.add_column(source2, style="yellow")

    info_table.add_row("Total Bars", f"{len(df1):,}", f"{len(df2):,}")
    info_table.add_row("First", str(df1["timestamp"].min())[:19], str(df2["timestamp"].min())[:19])
    info_table.add_row("Last", str(df1["timestamp"].max())[:19], str(df2["timestamp"].max())[:19])

    console.print(info_table)

    # Align timestamps (inner join)
    df1_renamed = df1.select([
        pl.col("timestamp"),
        pl.col("open").alias("open_1"),
        pl.col("high").alias("high_1"),
        pl.col("low").alias("low_1"),
        pl.col("close").alias("close_1"),
        pl.col("volume").alias("volume_1"),
    ])

    df2_renamed = df2.select([
        pl.col("timestamp"),
        pl.col("open").alias("open_2"),
        pl.col("high").alias("high_2"),
        pl.col("low").alias("low_2"),
        pl.col("close").alias("close_2"),
        pl.col("volume").alias("volume_2"),
    ])

    aligned = df1_renamed.join(df2_renamed, on="timestamp", how="inner")

    if len(aligned) == 0:
        console.print("\n[red]No overlapping timestamps between sources![/red]")
        console.print(f"  {source1}: {df1['timestamp'].min()!s} to {df1['timestamp'].max()!s}")
        console.print(f"  {source2}: {df2['timestamp'].min()!s} to {df2['timestamp'].max()!s}")
        raise typer.Exit(1)

    console.print(f"\n[green]Matched {len(aligned):,} bars with aligned timestamps[/green]")

    # Calculate differences
    diff_df = aligned.with_columns([
        (pl.col("open_1") - pl.col("open_2")).alias("open_diff"),
        (pl.col("high_1") - pl.col("high_2")).alias("high_diff"),
        (pl.col("low_1") - pl.col("low_2")).alias("low_diff"),
        (pl.col("close_1") - pl.col("close_2")).alias("close_diff"),
        (pl.col("volume_1") - pl.col("volume_2")).alias("volume_diff"),
    ])

    # Statistics table
    stats_table = Table(title="Difference Statistics")
    stats_table.add_column("Column", style="cyan")
    stats_table.add_column("Mean Diff", justify="right")
    stats_table.add_column("Max Diff", justify="right")
    stats_table.add_column("Std Dev", justify="right")
    stats_table.add_column("Correlation", justify="right")

    for col in ["open", "high", "low", "close"]:
        diff_col = f"{col}_diff"
        col1, col2 = f"{col}_1", f"{col}_2"

        mean_diff = diff_df[diff_col].mean()
        max_diff = diff_df[diff_col].abs().max()
        std_diff = diff_df[diff_col].std()

        # Calculate correlation
        corr = aligned.select(pl.corr(col1, col2)).item()

        stats_table.add_row(
            col.title(),
            f"{float(mean_diff):.6f}" if mean_diff is not None else "N/A",  # type: ignore[arg-type]
            f"{float(max_diff):.6f}" if max_diff is not None else "N/A",  # type: ignore[arg-type]
            f"{float(std_diff):.6f}" if std_diff is not None else "N/A",  # type: ignore[arg-type]
            f"{float(corr):.6f}" if corr is not None else "N/A",
        )

    # Volume stats (different scale)
    vol_mean = diff_df["volume_diff"].mean()
    vol_max = diff_df["volume_diff"].abs().max()
    vol_std = diff_df["volume_diff"].std()
    vol_corr = aligned.select(pl.corr("volume_1", "volume_2")).item()

    stats_table.add_row(
        "Volume",
        f"{float(vol_mean):,.0f}" if vol_mean is not None else "N/A",  # type: ignore[arg-type]
        f"{float(vol_max):,.0f}" if vol_max is not None else "N/A",  # type: ignore[arg-type]
        f"{float(vol_std):,.0f}" if vol_std is not None else "N/A",  # type: ignore[arg-type]
        f"{float(vol_corr):.6f}" if vol_corr is not None else "N/A",
    )

    console.print(stats_table)

    # Export if requested
    if output:
        output_path = Path(output)

        # Prepare export data
        statistics: dict[str, dict[str, float | None]] = {}
        for col in ["open", "high", "low", "close", "volume"]:
            diff_col = f"{col}_diff"
            col1, col2 = f"{col}_1", f"{col}_2"

            mean_diff = diff_df[diff_col].mean()
            max_diff = diff_df[diff_col].abs().max()
            std_diff = diff_df[diff_col].std()
            corr = aligned.select(pl.corr(col1, col2)).item()

            statistics[col] = {
                "mean_diff": float(mean_diff) if mean_diff is not None else None,  # type: ignore[arg-type]
                "max_diff": float(max_diff) if max_diff is not None else None,  # type: ignore[arg-type]
                "std_diff": float(std_diff) if std_diff is not None else None,  # type: ignore[arg-type]
                "correlation": float(corr) if corr is not None else None,
            }

        export_data: dict[str, object] = {
            "sources": {
                "source1": source1,
                "source2": source2,
            },
            "timeframe": timeframe,
            "matched_bars": len(aligned),
            "statistics": statistics,
        }

        if output_path.suffix.lower() == ".json":
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
        else:
            # CSV export - include the aligned data with differences
            diff_df.write_csv(output_path)

        console.print(f"\n[green]Exported to {output_path}[/green]")


@app.command("delete")
def delete_data(
    provider: Annotated[str, typer.Argument(help="Provider (e.g., 'oanda', 'binance')")],
    symbol: Annotated[str, typer.Argument(help="Symbol (e.g., 'EUR_USD')")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
) -> None:
    """Delete data for a specific provider/symbol/timeframe."""
    store = get_store()
    storage_key = get_storage_key(provider, symbol, timeframe)

    if not store.exists(storage_key):
        console.print(f"[yellow]No data found: {provider}/{symbol}/{timeframe}[/yellow]")
        raise typer.Exit(0)

    # Show what will be deleted
    console.print(f"\n[bold]Will delete:[/bold] {provider}/{symbol}/{timeframe}")
    console.print(f"  Storage key: {storage_key}\n")

    # Confirm unless --force
    if not force:
        confirm = typer.confirm("Are you sure you want to delete this data?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    # Delete via store
    store.delete(storage_key)
    console.print(f"[green]Deleted: {provider}/{symbol}/{timeframe}[/green]")


@app.command("validate-credentials")
def validate_credentials(provider: Annotated[str, typer.Argument(help="Provider name")]) -> None:
    """Validate provider credentials via DataService."""
    ds = DataService()
    ok = ds.validate_credentials(provider)
    console.print(f"Credentials valid: {ok}")


@app.command("gaps")
def gaps(
    provider: Annotated[str, typer.Argument(help="Provider name")],
    symbol: Annotated[str, typer.Argument(help="Symbol")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t")] = "1m",
    expected_minutes: Annotated[int, typer.Option("--expected-minutes")] = 1,
) -> None:
    """Show gaps for a symbol/timeframe."""
    ds = DataService()
    gap_list = ds.gaps(provider, symbol, timeframe, expected_minutes)
    if not gap_list:
        console.print("[green]No gaps detected[/green]")
        return
    for start, end in gap_list:
        console.print(f"{start} -> {end}")

"""Validation commands for liq-data CLI.

Commands for validating, auditing, and reporting on data quality.
"""

import logging
from datetime import timedelta
from typing import Annotated

import polars as pl
import typer
from rich.table import Table

from liq.data.cli.common import TIMEFRAME_MINUTES, console
from liq.data.settings import get_storage_key, get_store

logger = logging.getLogger(__name__)

# Weekend gap detection thresholds (in minutes)
# Forex markets close Friday ~5pm EST and reopen Sunday ~5pm EST (~48-72 hours)
MIN_WEEKEND_GAP_MINUTES = 2000  # ~33 hours minimum for weekend detection
MAX_WEEKEND_GAP_MINUTES = 5000  # ~83 hours maximum for weekend detection

app = typer.Typer()


@app.command("validate")
def validate_data(
    provider: Annotated[str, typer.Argument(help="Provider (e.g., 'oanda', 'binance')")],
    symbol: Annotated[str, typer.Argument(help="Symbol (e.g., 'EUR_USD')")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
) -> None:
    """Validate data integrity (gaps, nulls, duplicates, OHLC consistency)."""
    store = get_store()
    storage_key = get_storage_key(provider, symbol, timeframe)
    if not store.exists(storage_key):
        console.print(f"[red]Data not found: {provider}/{symbol}/{timeframe}[/red]")
        console.print(f"[yellow]Run: liq-data fetch {provider} {symbol} --timeframe {timeframe}[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]Validating: {provider}/{symbol}/{timeframe}[/bold blue]\n")

    df = store.read(storage_key)

    issues: list[tuple[str, str, str]] = []  # (severity, check, details)

    # Check for null values
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            pct = null_count / len(df) * 100
            issues.append(("WARNING", f"Null values in {col}", f"{null_count:,} ({pct:.2f}%)"))

    # Check for duplicates
    dup_count = len(df) - df.n_unique(subset=["timestamp"])
    if dup_count > 0:
        issues.append(("ERROR", "Duplicate timestamps", f"{dup_count:,} duplicates found"))

    # Check OHLC consistency (high >= low, high >= open/close, low <= open/close)
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        invalid_hl = df.filter(pl.col("high") < pl.col("low"))
        if len(invalid_hl) > 0:
            issues.append(("ERROR", "High < Low", f"{len(invalid_hl):,} bars"))

        invalid_ho = df.filter(pl.col("high") < pl.col("open"))
        if len(invalid_ho) > 0:
            issues.append(("ERROR", "High < Open", f"{len(invalid_ho):,} bars"))

        invalid_hc = df.filter(pl.col("high") < pl.col("close"))
        if len(invalid_hc) > 0:
            issues.append(("ERROR", "High < Close", f"{len(invalid_hc):,} bars"))

        invalid_lo = df.filter(pl.col("low") > pl.col("open"))
        if len(invalid_lo) > 0:
            issues.append(("ERROR", "Low > Open", f"{len(invalid_lo):,} bars"))

        invalid_lc = df.filter(pl.col("low") > pl.col("close"))
        if len(invalid_lc) > 0:
            issues.append(("ERROR", "Low > Close", f"{len(invalid_lc):,} bars"))

    # Check for negative values
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            neg_count = df.filter(pl.col(col) < 0).height
            if neg_count > 0:
                issues.append(("ERROR", f"Negative {col}", f"{neg_count:,} bars"))

    # Check timestamp ordering
    df_sorted = df.sort("timestamp")
    if not df["timestamp"].equals(df_sorted["timestamp"]):
        issues.append(("WARNING", "Timestamps not sorted", "Data is not in chronological order"))

    # Check for gaps (forex markets have gaps for weekends which is expected)
    expected_delta = timedelta(minutes=TIMEFRAME_MINUTES.get(timeframe, 1))

    df_with_diff = df_sorted.with_columns([
        (pl.col("timestamp").diff().dt.total_seconds() / 60).alias("gap_minutes")
    ])

    # Count gaps > 2x expected (allowing for some tolerance)
    max_expected = expected_delta.total_seconds() / 60 * 2
    large_gaps = df_with_diff.filter(pl.col("gap_minutes") > max_expected)

    # Filter out weekend gaps (Friday to Sunday/Monday)
    if len(large_gaps) > 0:
        # Weekend gaps are expected in forex
        weekend_gaps = large_gaps.filter(
            (pl.col("gap_minutes") > MIN_WEEKEND_GAP_MINUTES)
            & (pl.col("gap_minutes") < MAX_WEEKEND_GAP_MINUTES)
        )
        non_weekend_gaps = len(large_gaps) - len(weekend_gaps)
        if non_weekend_gaps > 0:
            issues.append(("INFO", "Unexpected gaps", f"{non_weekend_gaps:,} gaps > {max_expected:.0f} min"))

    # Summary table
    summary = Table(title="Validation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Total Bars", f"{len(df):,}")
    summary.add_row("Date Range", f"{df['timestamp'].min()!s} to {df['timestamp'].max()!s}")
    summary.add_row("Expected Timeframe", timeframe)
    summary.add_row("Unique Timestamps", f"{df.n_unique(subset=['timestamp']):,}")

    console.print(summary)

    # Issues table
    if issues:
        issues_table = Table(title="Validation Issues")
        issues_table.add_column("Severity", style="bold")
        issues_table.add_column("Check", style="cyan")
        issues_table.add_column("Details", style="yellow")

        for severity, check, details in issues:
            color = {"ERROR": "red", "WARNING": "yellow", "INFO": "blue"}.get(severity, "white")
            issues_table.add_row(f"[{color}]{severity}[/{color}]", check, details)

        console.print(issues_table)
    else:
        console.print("\n[green]✓ All validation checks passed![/green]")


@app.command("health-report")
def health_report(
    provider_filter: Annotated[
        str | None, typer.Option("--provider", "-p", help="Filter by provider")
    ] = None,
) -> None:
    """Show health report for all available data."""
    store = get_store()
    prefix = f"{provider_filter}/" if provider_filter else ""
    keys = store.list_keys(prefix=prefix)

    if not keys:
        filter_msg = f" for provider '{provider_filter}'" if provider_filter else ""
        console.print(f"[yellow]No data available{filter_msg}. Data directory is empty.[/yellow]")
        raise typer.Exit(0)

    console.print("\n[bold blue]Health Report[/bold blue]\n")

    # Build report data
    report_data: list[dict[str, str | int | float]] = []

    for key in sorted(keys):
        try:
            parts = key.split("/")
            if len(parts) < 3:
                continue

            prov = parts[0]
            sym = parts[1]
            tf = parts[2]

            df = store.read(key)
            date_range = store.get_date_range(key)

            # Calculate health metrics
            null_count = sum(df[col].null_count() for col in df.columns)
            dup_count = len(df) - df.n_unique(subset=["timestamp"])

            # Determine health status
            if dup_count > 0:
                status = "[red]Issues[/red]"
            elif null_count > 0:
                status = "[yellow]Warning[/yellow]"
            else:
                status = "[green]Healthy[/green]"

            report_data.append({
                "provider": prov,
                "symbol": sym,
                "timeframe": tf,
                "bars": len(df),
                "nulls": null_count,
                "dups": dup_count,
                "status": status,
                "first": str(date_range[0])[:10] if date_range else "N/A",
                "last": str(date_range[1])[:10] if date_range else "N/A",
            })

        except (pl.exceptions.ComputeError, pl.exceptions.SchemaError, OSError) as e:
            logger.debug("Skipping key %s: %s", key, e)
            continue

    if not report_data:
        filter_msg = f" for provider '{provider_filter}'" if provider_filter else ""
        console.print(f"[yellow]No data found{filter_msg}.[/yellow]")
        raise typer.Exit(0)

    # Display table
    table = Table(title=f"Health Report ({len(report_data)} datasets)")
    table.add_column("Provider", style="cyan")
    table.add_column("Symbol", style="green")
    table.add_column("TF", style="yellow")
    table.add_column("Bars", style="magenta", justify="right")
    table.add_column("First", style="dim")
    table.add_column("Last", style="dim")
    table.add_column("Status", style="bold")

    for row in report_data:
        table.add_row(
            str(row["provider"]),
            str(row["symbol"]),
            str(row["timeframe"]),
            f"{row['bars']:,}",
            str(row["first"]),
            str(row["last"]),
            str(row["status"]),
        )

    console.print(table)

    # Summary
    total_bars = sum(int(r["bars"]) for r in report_data)
    healthy = sum(1 for r in report_data if "Healthy" in str(r["status"]))

    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Total datasets: {len(report_data)}")
    console.print(f"  Total bars: {total_bars:,}")
    console.print(f"  Healthy: {healthy}/{len(report_data)}")


@app.command("audit")
def audit_data(
    provider: Annotated[str, typer.Argument(help="Provider (e.g., 'oanda', 'binance')")],
    symbol: Annotated[str, typer.Argument(help="Symbol (e.g., 'EUR_USD')")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be done")] = False,
) -> None:
    """Audit data quality: detect gaps, nulls, and anomalies."""
    store = get_store()
    storage_key = get_storage_key(provider, symbol, timeframe)

    if not store.exists(storage_key):
        console.print(f"[red]Data not found: {provider}/{symbol}/{timeframe}[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print(f"\n[bold blue]Audit (dry-run): {provider}/{symbol}/{timeframe}[/bold blue]")
        console.print(f"  Would analyze: {storage_key}")
        raise typer.Exit(0)

    console.print(f"\n[bold blue]Audit: {provider}/{symbol}/{timeframe}[/bold blue]\n")

    df = store.read(storage_key)

    # Quality metrics table
    quality_table = Table(title="Quality Metrics")
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Value", style="green")
    quality_table.add_column("Status", style="bold")

    total_bars = len(df)
    quality_table.add_row("Total Bars", f"{total_bars:,}", "[green]✓[/green]")

    # Check nulls
    null_counts = {col: df[col].null_count() for col in df.columns}
    total_nulls = sum(null_counts.values())
    null_status = "[green]✓[/green]" if total_nulls == 0 else "[yellow]![/yellow]"
    quality_table.add_row("Null Values", f"{total_nulls:,}", null_status)

    # Check duplicates
    dup_count = len(df) - df.n_unique(subset=["timestamp"])
    dup_status = "[green]✓[/green]" if dup_count == 0 else "[red]✗[/red]"
    quality_table.add_row("Duplicates", f"{dup_count:,}", dup_status)

    # Gap analysis
    expected_delta = timedelta(minutes=TIMEFRAME_MINUTES.get(timeframe, 1))

    df_sorted = df.sort("timestamp")
    df_with_diff = df_sorted.with_columns([
        (pl.col("timestamp").diff().dt.total_seconds() / 60).alias("gap_minutes")
    ])

    max_expected = expected_delta.total_seconds() / 60 * 2
    gaps = df_with_diff.filter(pl.col("gap_minutes") > max_expected)
    gap_count = len(gaps)
    gap_status = "[green]✓[/green]" if gap_count == 0 else "[yellow]![/yellow]"
    quality_table.add_row("Gaps Detected", f"{gap_count:,}", gap_status)

    console.print(quality_table)

    # Gap details if any
    if gap_count > 0:
        gap_table = Table(title="Gap Details (showing first 10)")
        gap_table.add_column("After Timestamp", style="cyan")
        gap_table.add_column("Gap (minutes)", style="yellow", justify="right")

        for row in gaps.head(10).iter_rows(named=True):
            gap_table.add_row(str(row["timestamp"]), f"{row['gap_minutes']:.0f}")

        console.print(gap_table)

    # Summary
    issues = []
    if total_nulls > 0:
        issues.append(f"{total_nulls} null values")
    if dup_count > 0:
        issues.append(f"{dup_count} duplicates")
    if gap_count > 0:
        issues.append(f"{gap_count} gaps")

    if issues:
        console.print(f"\n[yellow]Issues found: {', '.join(issues)}[/yellow]")
    else:
        console.print("\n[green]✓ Audit complete - no issues found[/green]")

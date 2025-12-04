"""Minimal Typer CLI for QA checks on bar data."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer
from rich.console import Console

from liq.data.qa import run_bar_qa

app = typer.Typer(help="liq-data QA utilities")
console = Console()


@app.command("qa")
def qa(
    bars_path: Path = typer.Argument(..., help="Path to bars in JSON or Parquet"),
) -> None:
    """Run bar-level QA checks and print summary."""
    if bars_path.suffix.lower() == ".parquet":
        df = pl.read_parquet(bars_path)
    else:
        with bars_path.open() as f:
            df = pl.from_dicts(json.load(f))
    if "timestamp" in df.columns:
        ts_dtype = df.schema["timestamp"]
        if not isinstance(ts_dtype, pl.Datetime):
            df = df.with_columns(
                pl.col("timestamp").str.to_datetime(time_unit="us", time_zone="UTC", strict=False)
            )
    qa_res = run_bar_qa(df)
    console.print(f"[cyan]Missing ratio:[/cyan] {qa_res.missing_ratio:.4f}")
    console.print(f"[cyan]Zero volume ratio:[/cyan] {qa_res.zero_volume_ratio:.4f}")
    console.print(f"[cyan]OHLC inconsistencies:[/cyan] {qa_res.ohlc_inconsistencies}")
    console.print(f"[cyan]Extreme moves:[/cyan] {qa_res.extreme_moves}")
    console.print(f"[cyan]Negative volume:[/cyan] {qa_res.negative_volume}")
    console.print(f"[cyan]Non-monotonic ts:[/cyan] {qa_res.non_monotonic_ts}")


if __name__ == "__main__":
    app()

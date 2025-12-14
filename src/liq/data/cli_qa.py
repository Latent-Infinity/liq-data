"""Minimal Typer CLI for QA checks on bar data.

All data access goes through liq-store for consistent storage abstraction.
"""

from __future__ import annotations

from typing import Annotated

import polars as pl
import typer
from rich.console import Console

from liq.data.qa import run_bar_qa
from liq.data.settings import get_store, get_storage_key

app = typer.Typer(help="liq-data QA utilities")
console = Console()


def _load_data(source: str) -> pl.DataFrame:
    """Load data from storage via liq-store.

    Args:
        source: Storage key (provider/symbol/bars/timeframe), e.g. oanda/EUR_USD/bars/1m

    Returns:
        DataFrame with bar data
    """
    parts = source.split("/")
    if len(parts) == 4 and parts[2] == "bars":
        provider, symbol, _, timeframe = parts
    elif len(parts) == 3:
        provider, symbol, timeframe = parts
    else:
        raise ValueError(
            "source must be a storage key of the form provider/symbol/bars/timeframe"
        )
    store = get_store()
    storage_key = get_storage_key(provider, symbol, timeframe)

    if not store.exists(storage_key):
        raise FileNotFoundError(f"Data not found: {storage_key}. Use liq-store-managed data.")

    return store.read(storage_key)


@app.command("qa")
def qa(
    source: Annotated[str, typer.Argument(help="Storage key: provider/symbol/bars/timeframe (e.g., oanda/EUR_USD/bars/1m)")],
) -> None:
    """Run bar-level QA checks and print summary via liq-store."""
    df = _load_data(source)

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

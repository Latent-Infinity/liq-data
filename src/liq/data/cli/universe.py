"""CLI subcommands for universe management + sync.

Subcommands:

* ``universe create`` — write a definition to the registry
* ``universe list`` — names of registered universes
* ``universe resolve`` — resolve as-of a date and print JSON
* ``universe delete`` — remove a definition (idempotent)
* ``sync`` — backfill / refresh a universe over a window

All write paths print JSON (single line) to stdout so callers can pipe.
Logs go to stderr via the existing ``console`` shared in ``cli.common``.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

import typer

from liq.data.cli.common import console, parse_date
from liq.data.service import DataService
from liq.data.settings import get_settings
from liq.data.universes import (
    UniverseConflictError,
    UniverseDefinition,
    UniverseKind,
    UniverseNotFoundError,
    UniverseRegistry,
    UniverseResolver,
)

universe_app = typer.Typer(
    help="Manage and resolve named universes.",
    no_args_is_help=True,
)


def _registry() -> UniverseRegistry:
    settings = get_settings()
    return UniverseRegistry(settings.data_root)


@universe_app.command("create")
def create_universe(
    name: Annotated[str, typer.Option("--name", "-n", help="Universe name (slug)")],
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            "-k",
            help="Kind: explicit | filter | composite | set_op",
        ),
    ],
    symbols: Annotated[
        str,
        typer.Option(
            "--symbols",
            "-s",
            help="Comma-separated symbols (kind=explicit)",
        ),
    ] = "",
    expr: Annotated[
        str,
        typer.Option("--expr", help="Polars boolean expression (kind=filter)"),
    ] = "",
    source: Annotated[
        str,
        typer.Option("--source", help="Constituent source name (kind=composite)"),
    ] = "",
    composite_id: Annotated[
        str,
        typer.Option("--id", help="Composite id, e.g. SP500 (kind=composite)"),
    ] = "",
    op: Annotated[
        str,
        typer.Option("--op", help="union | intersect | exclude (kind=set_op)"),
    ] = "",
    inputs: Annotated[
        str,
        typer.Option(
            "--inputs",
            help="Comma-separated input universe names (kind=set_op)",
        ),
    ] = "",
    version: Annotated[int, typer.Option("--version", "-v", help="Definition version")] = 1,
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Create a named universe and persist it to the registry."""
    try:
        spec = _build_spec(
            kind=kind,
            symbols=symbols,
            expr=expr,
            source=source,
            composite_id=composite_id,
            op=op,
            inputs=inputs,
        )
        definition = UniverseDefinition(
            name=name,
            version=version,
            kind=UniverseKind(kind),
            spec=spec,
        )
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    reg = _registry()
    try:
        path = reg.save(definition, overwrite=overwrite)
    except UniverseConflictError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    typer.echo(
        json.dumps(
            {
                "name": definition.name,
                "version": definition.version,
                "kind": definition.kind.value,
                "path": str(path),
            }
        )
    )


@universe_app.command("list")
def list_universes() -> None:
    """List the names of registered universes."""
    reg = _registry()
    typer.echo(json.dumps({"names": reg.list_names()}))


@universe_app.command("resolve")
def resolve_universe(
    name: Annotated[str, typer.Argument(help="Universe name")],
    as_of: Annotated[
        str,
        typer.Option("--as-of", help="ISO date for the resolution"),
    ],
) -> None:
    """Resolve a universe and print the symbols + PIT flag as JSON."""
    reg = _registry()
    try:
        definition = reg.load(name)
    except UniverseNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    as_of_date = parse_date(as_of)
    resolver = UniverseResolver(named_universes={n: reg.load(n) for n in reg.list_names()})
    resolved = resolver.resolve(definition, as_of_date)
    typer.echo(
        json.dumps(
            {
                "name": resolved.definition_name,
                "version": resolved.definition_version,
                "as_of": resolved.as_of.isoformat(),
                "pit": resolved.pit,
                "symbols": resolved.symbols,
            }
        )
    )


@universe_app.command("delete")
def delete_universe(
    name: Annotated[str, typer.Argument(help="Universe name")],
) -> None:
    """Remove a universe from the registry. Idempotent."""
    reg = _registry()
    removed = reg.delete(name)
    typer.echo(json.dumps({"name": name, "removed": removed}))


def sync_universe(
    universe_name: Annotated[str, typer.Argument(help="Universe name (looked up in the registry)")],
    start: Annotated[str, typer.Option("--start", "-s", help="Start date (YYYY-MM-DD)")],
    end: Annotated[str, typer.Option("--end", "-e", help="End date (YYYY-MM-DD)")] = "",
    provider: Annotated[str, typer.Option("--provider", help="Provider")] = "databento",
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe")] = "1m",
    dataset: Annotated[str, typer.Option("--dataset", help="Provider dataset code")] = "EQUS.MINI",
    force_refresh: bool = typer.Option(False, "--force-refresh"),
    budget_ack: bool = typer.Option(
        False,
        "--i-have-budget-authorization",
        help="Required when ``--force-refresh`` would re-bill the venue",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Stream per-symbol + per-batch progress events to stderr.",
    ),
    orchestration: str = typer.Option(
        "serial",
        "--orchestration",
        help="Execution mode: serial | batch",
    ),
    max_in_flight: int = typer.Option(
        4,
        "--max-in-flight",
        min=1,
        help="Maximum active provider batch jobs when --orchestration=batch.",
    ),
) -> None:
    """Sync a registered universe to local storage over ``[start, end]``."""
    if force_refresh and not budget_ack:
        console.print(
            "[red]--force-refresh re-fetches the full window from the venue "
            "(and re-bills). Pass --i-have-budget-authorization to confirm.[/red]"
        )
        raise typer.Exit(1)

    start_date = parse_date(start)
    end_date = parse_date(end) if end else date.today()

    if verbose:
        _install_sync_heartbeat()

    reg = _registry()
    service = DataService()
    try:
        if orchestration == "serial":
            report = service.sync(
                universe_name,
                start=start_date,
                end=end_date,
                provider=provider,
                timeframe=timeframe,
                dataset=dataset,
                force_refresh=force_refresh,
                registry=reg,
            )
        elif orchestration == "batch":
            report = service.sync_batch(
                universe_name,
                start=start_date,
                end=end_date,
                provider=provider,
                timeframe=timeframe,
                dataset=dataset,
                force_refresh=force_refresh,
                registry=reg,
                max_in_flight=max_in_flight,
            )
        else:
            console.print("[red]--orchestration must be one of: serial, batch[/red]")
            raise typer.Exit(1)
    except UniverseNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(report))


def _install_sync_heartbeat() -> None:
    """Attach a stderr handler that prints one short line per
    structured sync event. Idempotent — re-invocation is a no-op."""
    import logging

    root_logger = logging.getLogger("liq.data")
    root_logger.setLevel(logging.INFO)
    if any(getattr(h, "_liq_data_heartbeat", False) for h in root_logger.handlers):
        return

    class _HeartbeatHandler(logging.StreamHandler):
        _liq_data_heartbeat = True

        def emit(self, record: logging.LogRecord) -> None:
            event = getattr(record, "event", None)
            if event is None:
                return
            symbol = getattr(record, "symbol", "")
            job_id = getattr(record, "job_id", "")
            state = getattr(record, "state", "")
            extras = " ".join(
                f"{k}={v}"
                for k, v in (("symbol", symbol), ("job_id", job_id), ("state", state))
                if v
            )
            try:
                self.stream.write(f"[{event}] {extras}\n".strip() + "\n")
                self.stream.flush()
            except Exception:  # noqa: BLE001 — heartbeat must never crash the sync
                pass

    import sys

    handler = _HeartbeatHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)


# ----- helpers --------------------------------------------------------------


def _build_spec(
    *,
    kind: str,
    symbols: str,
    expr: str,
    source: str,
    composite_id: str,
    op: str,
    inputs: str,
) -> dict:
    if kind == "explicit":
        if not symbols:
            raise ValueError("--symbols is required for kind=explicit")
        return {"symbols": [s.strip() for s in symbols.split(",") if s.strip()]}
    if kind == "filter":
        if not expr:
            raise ValueError("--expr is required for kind=filter")
        return {"expr": expr}
    if kind == "composite":
        if not (source and composite_id):
            raise ValueError("--source and --id are required for kind=composite")
        return {"source": source, "id": composite_id}
    if kind == "set_op":
        if not (op and inputs):
            raise ValueError("--op and --inputs are required for kind=set_op")
        return {
            "op": op,
            "inputs": [s.strip() for s in inputs.split(",") if s.strip()],
        }
    raise ValueError(f"unknown universe kind: {kind!r}")

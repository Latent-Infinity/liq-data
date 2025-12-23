"""Main CLI module for liq-data.

This module assembles all CLI submodules into a single application.
"""

import typer

from liq.data.cli import auth, fetch, info, manage, validate

app = typer.Typer(
    name="liq-data",
    help="Market data fetching CLI for the LIQ Stack",
    no_args_is_help=True,
)

# Register commands from submodules
# Fetch commands
app.command("fetch")(fetch.fetch_bars)
app.command("backfill")(fetch.backfill_bars)

# Info commands
app.command("list")(info.list_instruments)
app.command("config")(info.show_config)
app.command("info")(info.show_data_info)
app.command("stats")(info.show_stats)

# Validation commands
app.command("validate")(validate.validate_data)
app.command("health-report")(validate.health_report)
app.command("audit")(validate.audit_data)

# Data management commands
app.command("compare")(manage.compare_data)
app.command("delete")(manage.delete_data)

# Auth helpers
app.command("tradestation-auth-url")(auth.tradestation_auth_url)
app.command("tradestation-exchange-code")(auth.tradestation_exchange_code)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

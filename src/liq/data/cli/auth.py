"""Authentication helpers for provider-specific flows."""

from typing import Annotated
import secrets

import typer

from liq.data.cli.common import console
from liq.data.exceptions import AuthenticationError
from liq.data.providers.tradestation import TradeStationProvider
from liq.data.settings import get_settings

app = typer.Typer(help="Authentication helpers for provider flows")


@app.command("tradestation-auth-url")
def tradestation_auth_url(
    redirect_uri: Annotated[str | None, typer.Option("--redirect-uri", help="OAuth redirect URI")] = None,
    scope: Annotated[str | None, typer.Option("--scope", help="OAuth scopes")] = None,
    audience: Annotated[str | None, typer.Option("--audience", help="OAuth audience")] = None,
    state: Annotated[str | None, typer.Option("--state", help="Opaque state to prevent CSRF")] = None,
) -> None:
    """Generate the TradeStation authorization URL for the Auth Code flow."""
    settings = get_settings()
    client_id = settings.tradestation_client_id
    if not client_id:
        console.print("[red]TRADESTATION_CLIENT_ID not configured.[/red]")
        raise typer.Exit(1)

    redirect = redirect_uri or settings.tradestation_redirect_uri
    if not redirect:
        console.print("[red]Missing redirect URI. Set TRADESTATION_REDIRECT_URI or pass --redirect-uri.[/red]")
        raise typer.Exit(1)

    resolved_scope = scope or settings.tradestation_scopes
    resolved_state = state or secrets.token_urlsafe(16)

    url = TradeStationProvider.build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect,
        scope=resolved_scope,
        state=resolved_state,
        audience=audience,
    )

    console.print("[green]Authorization URL generated.[/green]")
    console.print(f"[bold]State:[/bold] {resolved_state}")
    console.print(url)


@app.command("tradestation-exchange-code")
def tradestation_exchange_code(
    code: Annotated[str, typer.Argument(help="Authorization code from redirect")],
    redirect_uri: Annotated[str | None, typer.Option("--redirect-uri", help="OAuth redirect URI")] = None,
) -> None:
    """Exchange a TradeStation auth code for a refresh token."""
    settings = get_settings()
    client_id = settings.tradestation_client_id
    client_secret = settings.tradestation_client_secret
    if not client_id or not client_secret:
        console.print("[red]TRADESTATION_CLIENT_ID/SECRET not configured.[/red]")
        raise typer.Exit(1)

    redirect = redirect_uri or settings.tradestation_redirect_uri
    if not redirect:
        console.print("[red]Missing redirect URI. Set TRADESTATION_REDIRECT_URI or pass --redirect-uri.[/red]")
        raise typer.Exit(1)

    try:
        data = TradeStationProvider.exchange_authorization_code(
            client_id=client_id,
            client_secret=client_secret,
            code=code,
            redirect_uri=redirect,
        )
    except AuthenticationError as exc:
        console.print(f"[red]Auth code exchange failed: {exc}[/red]")
        raise typer.Exit(1)

    refresh_token = data.get("refresh_token")
    if not refresh_token:
        console.print("[red]No refresh token returned. Ensure offline_access scope is included.[/red]")
        raise typer.Exit(1)

    console.print("[green]Refresh token received.[/green]")
    console.print("Set this in your .env as TRADESTATION_REFRESH_TOKEN:")
    console.print(refresh_token)

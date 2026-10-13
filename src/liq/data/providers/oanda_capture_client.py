"""Provider-local capture client policy; guarded callers retain one lifecycle."""

from typing import Literal

import httpx

from liq.data.providers.oanda import OandaProvider


class CaptureCloseRefused(PermissionError):
    """Category-only ordinary owned-close failure; no underlying exception text."""

    def __init__(self) -> None:
        self.reason: Literal["CLOSE_REFUSED"] = "CLOSE_REFUSED"
        super().__init__(self.reason)


class CaptureOwnerRefused(PermissionError):
    """Category-only client ownership or configuration refusal."""

    def __init__(self) -> None:
        self.reason: Literal["CLIENT_POLICY_REFUSED"] = "CLIENT_POLICY_REFUSED"
        super().__init__(self.reason)


def _provider_identity(provider: OandaProvider) -> tuple[str, str, str, str, float]:
    return (
        provider._api_key,
        provider._account_id,
        provider._environment,
        provider._base_url,
        provider._timeout,
    )


class CaptureClientOwner:
    """One genuine client and internal transport; never a request executor.

    A guarded service owner must validate its account binding before installing.
    This component alone supplies neither guard authority nor a capture permit.
    """

    def __init__(self) -> None:
        self._provider: OandaProvider | None = None
        self._client: httpx.Client | None = None
        self._transport: httpx.HTTPTransport | None = None
        self._identity: tuple[str, str, str, str, float] | None = None
        self._headers: tuple[tuple[bytes, bytes], ...] = ()
        self._closed = False
        self._close_failed = False

    def install(self, provider: OandaProvider) -> None:
        """Install once into a fresh provider, after caller's guard and binding."""
        if (
            self._closed
            or self._provider is not None
            or type(provider) is not OandaProvider
            or provider._closed
            or provider._client is not None
            or provider._environment != "live"
            or provider._base_url != OandaProvider.LIVE_URL
            or provider._timeout != 30.0
        ):
            raise CaptureOwnerRefused()
        self._provider = provider
        self._identity = _provider_identity(provider)
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {provider._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
            trust_env=False,
            follow_redirects=False,
        )
        self._headers = tuple(self._client.headers.raw)
        transport = self._client._transport
        if type(transport) is not httpx.HTTPTransport:
            raise CaptureOwnerRefused()
        self._transport = transport
        provider._client = self._client
        self.revalidate(provider)

    def revalidate(self, provider: OandaProvider) -> None:
        """Check retained identities and lifecycle without calling the SDK getter."""
        if self._close_failed:
            raise CaptureCloseRefused()
        client = self._client
        transport = self._transport
        if (
            provider is not self._provider
            or _provider_identity(provider) != self._identity
            or client is None
            or transport is None
            or provider._client is not client
            or client._transport is not transport
            or provider._closed != self._closed
            or client.is_closed != self._closed
            or not self._strict_policy(client, transport, provider)
        ):
            raise CaptureOwnerRefused()

    def _strict_policy(
        self, client: httpx.Client, transport: httpx.HTTPTransport, provider: OandaProvider
    ) -> bool:
        return (
            type(client) is httpx.Client
            and type(transport) is httpx.HTTPTransport
            and client.trust_env is False
            and client._mounts == {}
            and getattr(transport._pool, "_retries", None) == 0
            and client.auth is None
            and client.event_hooks == {"request": [], "response": []}
            and not client.follow_redirects
            and client.timeout == httpx.Timeout(30.0)
            and not client.params
            and len(client.cookies) == 0
            and tuple(client.headers.raw) == self._headers
            and client.headers.get_list("Authorization") == [f"Bearer {provider._api_key}"]
            and client.headers.get_list("Content-Type") == ["application/json"]
        )

    def close(self) -> None:
        """Close only owned resources once; ordinary failure never exposes context."""
        if self._closed:
            if self._close_failed:
                raise CaptureCloseRefused()
            return
        provider = self._provider
        client = self._client
        drift = False
        if provider is not None and client is not None:
            try:
                self.revalidate(provider)
            except CaptureOwnerRefused:
                drift = True
        self._closed = True
        self._close_failed = True
        try:
            if provider is not None:
                if drift and client is not None:
                    provider._closed = True
                    if not client.is_closed:
                        if self._transport is not None:
                            client._transport = self._transport
                            client._mounts = {}
                        client.close()
                else:
                    provider.close()
        except Exception:
            pass
        else:
            self._close_failed = drift
        if self._close_failed:
            raise CaptureCloseRefused()

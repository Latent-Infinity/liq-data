"""Mechanical lifecycle checks; no API requests or market-data payloads."""

from collections.abc import Iterator

import httpx
import pytest

from liq.data.exceptions import ProviderError
from liq.data.providers.oanda import OandaProvider


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[OandaProvider]:
    instance = OandaProvider("dummy-token", "dummy-account")
    yield instance
    monkeypatch.undo()
    instance.__del__()


def test_close_before_client_creation_is_terminal(
    provider: OandaProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid_creation(*args: object, **kwargs: object) -> httpx.Client:
        pytest.fail("closed provider allocated a client")

    monkeypatch.setattr(httpx, "Client", forbid_creation)
    provider.close()
    provider.close()
    with pytest.raises(ProviderError, match="^PROVIDER_CLOSED$"):
        provider._get_client()
    assert provider._client is None


def test_close_keeps_identity_but_never_reopens(
    provider: OandaProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = provider._get_client()
    assert provider._get_client() is client
    closes: list[httpx.Client] = []
    original_close = httpx.Client.close

    def tracked_close(instance: httpx.Client) -> None:
        closes.append(instance)
        original_close(instance)

    monkeypatch.setattr(httpx.Client, "close", tracked_close)
    provider.close()
    provider.close()
    provider.__del__()
    assert closes == [client]
    assert client.is_closed
    assert provider._client is client
    with pytest.raises(ProviderError, match="^PROVIDER_CLOSED$"):
        provider._get_client()


def test_failed_transport_shutdown_is_not_retried(
    provider: OandaProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = httpx.Client(trust_env=False)
    provider._client = client
    closes: list[httpx.HTTPTransport] = []
    original_close = httpx.HTTPTransport.close

    def failed_close(transport: httpx.HTTPTransport) -> None:
        closes.append(transport)
        original_close(transport)
        raise OSError("dummy private shutdown detail")

    monkeypatch.setattr(httpx.HTTPTransport, "close", failed_close)
    with pytest.raises(OSError, match="dummy private shutdown detail"):
        provider.close()
    provider.close()
    provider.__del__()
    assert len(closes) == 1
    assert client.is_closed
    assert provider._client is client
    with pytest.raises(ProviderError, match="^PROVIDER_CLOSED$"):
        provider._get_client()


def test_destructor_handles_incomplete_initialization() -> None:
    instance = object.__new__(OandaProvider)
    instance.__del__()
    assert not hasattr(instance, "_client")


def test_shutdown_marks_terminal_before_touching_client(
    provider: OandaProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = provider._get_client()
    original_close = httpx.Client.close

    def check_terminal(instance: httpx.Client) -> None:
        with pytest.raises(ProviderError, match="^PROVIDER_CLOSED$"):
            provider._get_client()
        original_close(instance)

    monkeypatch.setattr(httpx.Client, "close", check_terminal)
    provider.close()
    assert client.is_closed


def test_destructor_closes_unclosed_provider_once(
    provider: OandaProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = provider._get_client()
    closes: list[httpx.Client] = []
    original_close = httpx.Client.close

    def tracked_close(instance: httpx.Client) -> None:
        closes.append(instance)
        original_close(instance)

    monkeypatch.setattr(httpx.Client, "close", tracked_close)
    provider.__del__()
    provider.__del__()
    assert closes == [client]
    assert client.is_closed


def test_unclosed_provider_preserves_connection_settings(provider: OandaProvider) -> None:
    client = provider._get_client()
    assert provider._base_url == OandaProvider.PRACTICE_URL
    assert client.headers["Authorization"] == "Bearer dummy-token"
    assert client.headers["Content-Type"] == "application/json"
    assert client.timeout == httpx.Timeout(30.0)
    assert not client.is_closed
    assert provider._get_client() is client

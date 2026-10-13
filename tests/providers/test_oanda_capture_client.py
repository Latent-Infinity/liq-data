"""Canonical capture-client capability stays isolated from experiment packages."""

import httpx
import pytest

from liq.data.providers.oanda import OandaProvider, OandaProviderClosed
from liq.data.providers.oanda_capture_client import CaptureClientOwner, CaptureOwnerRefused


def test_capture_owner_installs_one_genuine_client_without_environment_routing() -> None:
    provider = OandaProvider("dummy-token", "dummy-account", environment="live")
    owner = CaptureClientOwner()
    try:
        owner.install(provider)
        owner.revalidate(provider)
        assert type(provider._client) is httpx.Client
        assert provider._client.trust_env is False and provider._client._mounts == {}
        assert provider._client.follow_redirects is False
        assert provider._client is owner._client
    finally:
        owner.close()


def test_capture_owned_close_terminalizes_sdk_before_any_reacquisition() -> None:
    provider = OandaProvider("dummy-token", "dummy-account", environment="live")
    owner = CaptureClientOwner()
    owner.install(provider)
    owner.close()
    owner.close()
    with pytest.raises(OandaProviderClosed):
        provider._get_client()
    with pytest.raises(CaptureOwnerRefused):
        owner.install(provider)
    assert provider._client is owner._client
    assert provider._closed is True and provider._client is not None and provider._client.is_closed

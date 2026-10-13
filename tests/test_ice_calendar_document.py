"""Source-free controls for one fixed documentary transport, without admission."""

import httpx
import pytest

from liq.data.providers.ice_calendar_document import ICE_URL, capture_ice_calendar


@pytest.mark.parametrize(
    "status,headers",
    [
        (403, {}),
        (302, {"location": "https://example.org"}),
        (200, {"content-encoding": "gzip"}),
        (200, {"content-length": "4194304"}),
        (200, {"content-length": "-1"}),
        (200, {"content-length": "invalid"}),
    ],
)
def test_fixed_document_refuses_without_retry(status, headers) -> None:
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(status, headers=headers, content=b"")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client, pytest.raises(ValueError):
        capture_ice_calendar(client)
    assert len(seen) == 1
    assert str(seen[0].url) == ICE_URL
    assert seen[0].headers["accept-encoding"] == "identity"


def test_fixed_document_returns_only_original_bytes() -> None:
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"<p>documentary control</p>")
        )
    ) as client:
        assert capture_ice_calendar(client) == b"<p>documentary control</p>"


def test_streamed_document_enforces_byte_limit() -> None:
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, stream=httpx.ByteStream(b"x" * 4194304))
            )
        ) as client,
        pytest.raises(ValueError, match="byte_budget"),
    ):
        capture_ice_calendar(client)


def test_owned_client_disallows_redirects_and_uses_one_request(monkeypatch) -> None:
    original_client = httpx.Client
    seen = []

    def handle(request):
        seen.append(request)
        assert request.extensions["timeout"] == {
            "connect": 20.0,
            "read": 20.0,
            "write": 20.0,
            "pool": 20.0,
        }
        return httpx.Response(200, content=b"documentary control")

    def factory(**kwargs):
        assert kwargs == {"follow_redirects": False}
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr("liq.data.providers.ice_calendar_document.httpx.Client", factory)
    assert capture_ice_calendar() == b"documentary control"
    assert len(seen) == 1

"""Historical-revision transport controls; opaque bytes are not constituent evidence."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Final

import httpx
import pytest
import respx

from liq.data.exceptions import ConfigurationError
from liq.data.providers.sp500_membership import WIKIPEDIA_API_URL, SP500MembershipProvider

CUTOFF: Final = datetime(2023, 4, 21, tzinfo=UTC)
LIMIT: Final = 64 * 1024
PARAMS: Final = {
    "action": "query",
    "prop": "revisions",
    "titles": "List of S&P 500 companies",
    "rvlimit": "1",
    "rvstart": "2023-04-21T00:00:00Z",
    "rvdir": "older",
    "rvprop": "ids|timestamp|size|sha1",
    "format": "json",
    "formatversion": "2",
    "maxlag": "1",
}


class OpaqueStream(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b"x" * (LIMIT // 2)
        yield b"x" * (LIMIT // 2)


@respx.mock
def test_revision_capture_requests_one_historical_page_and_preserves_raw_bytes() -> None:
    raw = b"not-json; must be retained before any decoding"
    route = respx.get(WIKIPEDIA_API_URL, params=PARAMS).mock(
        return_value=httpx.Response(200, content=raw)
    )
    provider = SP500MembershipProvider(user_agent="protocol-test test@example.com")
    try:
        assert provider.capture_wikipedia_revision_reference(CUTOFF) == raw
    finally:
        provider.close()
    assert len(respx.calls) == route.call_count == 1
    request = route.calls[0].request
    assert dict(request.url.params) == PARAMS
    assert request.headers["Accept-Encoding"] == "identity"
    assert set(request.extensions["timeout"].values()) == {20.0}


@respx.mock
@pytest.mark.parametrize(
    "cutoff",
    [
        datetime(2023, 4, 21),
        datetime(2023, 4, 21, tzinfo=timezone(timedelta(hours=1))),
        datetime(2023, 4, 21, microsecond=1, tzinfo=UTC),
    ],
)
def test_revision_capture_rejects_ambiguous_cutoff_before_transport(cutoff: datetime) -> None:
    provider = SP500MembershipProvider(user_agent="protocol-test test@example.com")
    try:
        with pytest.raises(ConfigurationError, match="UTC"):
            provider.capture_wikipedia_revision_reference(cutoff)
    finally:
        provider.close()
    assert not respx.calls


@respx.mock
@pytest.mark.parametrize("status", [301, 302, 403, 429, 500])
def test_revision_capture_never_follows_redirect_or_retries(status: int) -> None:
    route = respx.get(WIKIPEDIA_API_URL, params=PARAMS).mock(
        return_value=httpx.Response(status, headers={"Location": "https://example.com/unused"})
    )
    provider = SP500MembershipProvider(user_agent="protocol-test test@example.com")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            provider.capture_wikipedia_revision_reference(CUTOFF)
    finally:
        provider.close()
    assert len(respx.calls) == route.call_count == 1


@respx.mock
@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": str(LIMIT)},
        {"Content-Length": "-1"},
        {"Content-Length": "invalid"},
        {"Content-Encoding": "gzip"},
    ],
)
def test_revision_capture_refuses_invalid_or_oversized_transport_metadata(
    headers: dict[str, str],
) -> None:
    respx.get(WIKIPEDIA_API_URL, params=PARAMS).mock(
        return_value=httpx.Response(200, headers=headers)
    )
    provider = SP500MembershipProvider(user_agent="protocol-test test@example.com")
    try:
        with pytest.raises(ValueError):
            provider.capture_wikipedia_revision_reference(CUTOFF)
    finally:
        provider.close()


@respx.mock
def test_revision_capture_limits_stream_without_a_content_length() -> None:
    respx.get(WIKIPEDIA_API_URL, params=PARAMS).mock(
        return_value=httpx.Response(200, stream=OpaqueStream())
    )
    provider = SP500MembershipProvider(user_agent="protocol-test test@example.com")
    try:
        with pytest.raises(ValueError, match="budget"):
            provider.capture_wikipedia_revision_reference(CUTOFF)
    finally:
        provider.close()

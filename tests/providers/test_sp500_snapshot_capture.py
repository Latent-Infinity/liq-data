"""Bounded reference transport controls, without constituent observations."""

import httpx
import pytest
import respx

from liq.data.providers.sp500_membership import SNAPSHOT_CSV_URL, SP500MembershipProvider

REVISION = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
URL = SNAPSHOT_CSV_URL.replace("/master/", f"/{REVISION}/")


@respx.mock
def test_date_query_notebook_capture_only_fetches_pinned_document() -> None:
    url = f"https://raw.githubusercontent.com/fja05680/sp500/{REVISION}/sp500_by_date.ipynb"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=b"{}"))
    provider = SP500MembershipProvider(user_agent="fixture test@example.com")
    try:
        assert provider.capture_date_query_reference(REVISION) == b"{}"
    finally:
        provider.close()
    assert route.call_count == 1
    assert len(respx.calls) == 1


@respx.mock
def test_snapshot_capture_uses_immutable_revision_without_redirects() -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"date,tickers\n"))
    provider = SP500MembershipProvider(user_agent="fixture test@example.com")
    try:
        assert provider.capture_snapshot_reference(REVISION) == b"date,tickers\n"
    finally:
        provider.close()
    assert route.call_count == 1
    assert len(respx.calls) == 1
    assert route.calls[0].request.headers["Accept-Encoding"] == "identity"


@respx.mock
@pytest.mark.parametrize("revision", ["master", "../main", "a" * 39, "A" * 40])
def test_snapshot_capture_refuses_noncanonical_revision_before_request(revision: str) -> None:
    provider = SP500MembershipProvider(user_agent="fixture test@example.com")
    with pytest.raises(ValueError, match="revision"):
        provider.capture_snapshot_reference(revision)
    assert not respx.calls


@respx.mock
@pytest.mark.parametrize("status", [301, 403, 429, 500])
def test_snapshot_capture_never_retries_or_follows_redirect(status: int) -> None:
    route = respx.get(URL).mock(
        return_value=httpx.Response(status, headers={"Location": "https://example.com/forbidden"})
    )
    provider = SP500MembershipProvider(user_agent="fixture test@example.com")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            provider.capture_snapshot_reference(REVISION)
    finally:
        provider.close()
    assert route.call_count == 1


@respx.mock
def test_snapshot_capture_refuses_declared_body_at_ceiling() -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(200, headers={"Content-Length": str(8 * 1024 * 1024)})
    )
    provider = SP500MembershipProvider(user_agent="fixture test@example.com")
    try:
        with pytest.raises(ValueError, match="budget"):
            provider.capture_snapshot_reference(REVISION)
    finally:
        provider.close()


@respx.mock
def test_snapshot_capture_refuses_stream_at_ceiling_without_declared_size() -> None:
    respx.get(URL).mock(
        return_value=httpx.Response(
            200, content=b" " * (8 * 1024 * 1024), headers={"Content-Length": "0"}
        )
    )
    provider = SP500MembershipProvider(user_agent="fixture test@example.com")
    try:
        with pytest.raises(ValueError, match="budget"):
            provider.capture_snapshot_reference(REVISION)
    finally:
        provider.close()


@respx.mock
@pytest.mark.parametrize(
    "headers",
    [{"Content-Encoding": "identity, gzip"}, {"Content-Length": "bad"}, {"Content-Length": "-1"}],
)
def test_snapshot_capture_refuses_invalid_transport_metadata(headers: dict[str, str]) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, headers=headers))
    provider = SP500MembershipProvider(user_agent="fixture test@example.com")
    try:
        with pytest.raises(ValueError):
            provider.capture_snapshot_reference(REVISION)
    finally:
        provider.close()

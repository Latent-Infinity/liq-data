"""Bounded SEC reference transport controls; no financial fixtures."""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from liq.data.exceptions import ProviderError
from liq.data.providers.sec_edgar import TICKERS_URL, SECEdgarProvider, _parse_acceptance


def test_acceptance_timestamp_requires_explicit_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _parse_acceptance("2023-02-01T16:00:00")


@respx.mock
def test_filing_capture_reads_only_the_named_document() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1637459/000163745923000005/khc-20230215.htm"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=b"<html></html>"))
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    assert (
        provider.capture_filing_document("0001637459", "0001637459-23-000005", "khc-20230215.htm")
        == b"<html></html>"
    )
    assert route.call_count == 1
    assert len(respx.calls) == 1


@respx.mock
def test_filing_capture_uses_declared_smaller_source_budget() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1637459/000163745922000056/khc-20220427.htm"
    route = respx.get(url).mock(return_value=httpx.Response(200, content=b"NONEXECUTING_METADATA"))
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    assert (
        provider.capture_filing_document(
            "0001637459", "0001637459-22-000056", "khc-20220427.htm", max_source_bytes=1024
        )
        == b"NONEXECUTING_METADATA"
    )
    assert route.call_count == 1


@respx.mock
def test_filing_capture_stops_at_smaller_stream_ceiling() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1637459/000163745922000056/khc-20220427.htm"
    route = respx.get(url).mock(
        return_value=httpx.Response(200, content=b" " * 1024, headers={"Content-Length": "0"})
    )
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ProviderError, match="byte budget"):
        provider.capture_filing_document(
            "0001637459", "0001637459-22-000056", "khc-20220427.htm", max_source_bytes=1024
        )
    assert route.call_count == 1


@respx.mock
def test_filing_capture_rejects_declared_size_at_smaller_ceiling() -> None:
    url = "https://www.sec.gov/Archives/edgar/data/1637459/000163745922000056/khc-20220427.htm"
    route = respx.get(url).mock(
        return_value=httpx.Response(200, headers={"Content-Length": "1024"})
    )
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ProviderError, match="byte budget"):
        provider.capture_filing_document(
            "0001637459", "0001637459-22-000056", "khc-20220427.htm", max_source_bytes=1024
        )
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize("limit", [0, 1023, 1025, 4 * 1024 * 1024 + 1024, True])
def test_filing_capture_invalid_byte_budget_refuses_before_request(limit: int) -> None:
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ValueError, match="source byte budget"):
        provider.capture_filing_document(
            "0001637459", "0001637459-22-000056", "khc-20220427.htm", max_source_bytes=limit
        )
    assert len(respx.calls) == 0


@respx.mock
def test_filing_capture_refuses_path_traversal_before_request() -> None:
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ValueError, match="basename"):
        provider.capture_filing_document("0001637459", "0001637459-23-000005", "../other.htm")
    assert len(respx.calls) == 0


def test_acceptance_timestamp_normalizes_explicit_offset() -> None:
    assert _parse_acceptance("2023-02-01T16:00:00-05:00") == datetime(2023, 2, 1, 21, tzinfo=UTC)


@respx.mock
def test_reference_capture_is_one_identity_encoded_request() -> None:
    route = respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, content=b"{}"))
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    assert provider.capture_ticker_reference() == b"{}"
    assert route.call_count == 1
    assert route.calls[0].request.headers["Accept-Encoding"] == "identity"


@respx.mock
@pytest.mark.parametrize("status", [301, 403, 429, 500])
def test_reference_capture_never_retries_or_redirects(status: int) -> None:
    route = respx.get(TICKERS_URL).mock(
        return_value=httpx.Response(status, headers={"Location": "https://example.com/forbidden"})
    )
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ProviderError, match="reference capture"):
        provider.capture_ticker_reference()
    assert route.call_count == 1


@respx.mock
def test_reference_capture_refuses_compressed_response() -> None:
    respx.get(TICKERS_URL).mock(
        return_value=httpx.Response(200, headers={"Content-Encoding": "identity, gzip"})
    )
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ProviderError, match="encoding"):
        provider.capture_ticker_reference()


@respx.mock
def test_reference_capture_refuses_declared_oversize_without_reading() -> None:
    respx.get(TICKERS_URL).mock(
        return_value=httpx.Response(200, headers={"Content-Length": str(4 * 1024 * 1024 + 1)})
    )
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ProviderError, match="byte budget"):
        provider.capture_ticker_reference()


@respx.mock
def test_reference_capture_refuses_stream_at_byte_ceiling() -> None:
    respx.get(TICKERS_URL).mock(
        return_value=httpx.Response(
            200, content=b" " * (4 * 1024 * 1024), headers={"Content-Length": "0"}
        )
    )
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ProviderError, match="byte budget"):
        provider.capture_ticker_reference()


@respx.mock
def test_root_submissions_capture_is_one_exact_object() -> None:
    route = respx.get("https://data.sec.gov/submissions/CIK0001637459.json").mock(
        return_value=httpx.Response(200, content=b"{}")
    )
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    assert provider.capture_submissions_index("0001637459") == b"{}"
    assert route.call_count == 1
    assert len(respx.calls) == 1


@respx.mock
@pytest.mark.parametrize("cik", ["1637459", "../0001637459", "0000000000", "0001637459.json"])
def test_root_submissions_capture_refuses_noncanonical_identity(cik: str) -> None:
    provider = SECEdgarProvider(user_agent="fixture test@example.com", min_interval_seconds=0)
    with pytest.raises(ValueError, match="CIK"):
        provider.capture_submissions_index(cik)
    assert len(respx.calls) == 0

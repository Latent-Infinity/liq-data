"""Non-financial wire fixtures exercise bounds, not native-price admission."""

from collections.abc import Iterator
from hashlib import sha256

import httpx
import pytest

from liq.data.providers.oanda_source_body import (
    MAX_SOURCE_BODY_BYTES,
    SourceBody,
    SourceBodyFailure,
    read_source_body,
)


class WireBody(httpx.SyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...], fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.reads = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.reads += 1
            yield chunk
        if self.fail:
            raise httpx.ReadError("private-account/token must never enter a receipt")

    def close(self) -> None:
        self.closed = True


def response_for(body: WireBody, headers: list[tuple[str, str]]) -> httpx.Response:
    return httpx.Response(200, headers=headers, stream=body)


@pytest.mark.parametrize("headers", [[], [("Content-Encoding", "identity")]])
def test_complete_identity_body_preserves_exact_entity_bytes(
    headers: list[tuple[str, str]],
) -> None:
    # Given non-financial JSON split across raw transfer chunks.
    body = WireBody((b'{ "transport": ', b'"only" }\n'))
    response = response_for(body, headers)
    # When the bounded reader completes.
    result = read_source_body(response)
    # Then bytes are neither decoded nor reserialized and the response is closed.
    assert isinstance(result, SourceBody)
    assert result.body == b'{ "transport": "only" }\n'
    assert result.sha256 == sha256(result.body).hexdigest()
    assert "transport" not in repr(result)
    assert body.closed and response.is_closed


@pytest.mark.parametrize(
    "headers",
    [
        [("Content-Encoding", "gzip")],
        [("Content-Encoding", "Identity")],
        [("Content-Encoding", "")],
        [("Content-Encoding", "identity"), ("Content-Encoding", "identity")],
        [("Content-Encoding", "identity, gzip")],
    ],
)
def test_encoding_refused_before_read(headers: list[tuple[str, str]]) -> None:
    body = WireBody((b"unread",))
    result = read_source_body(response_for(body, headers))
    assert result == SourceBodyFailure("ENCODING_REFUSED", 200, 0)
    assert body.reads == 0 and body.closed


@pytest.mark.parametrize(
    "headers",
    [
        [("Content-Length", "-1")],
        [("Content-Length", "+1")],
        [("Content-Length", "1.0")],
        [("Content-Length", "")],
        [("Content-Length", " 1")],
        [("Content-Length", "1 ")],
        [("Content-Length", "1,1")],
        [("Content-Length", "1"), ("Content-Length", "1")],
    ],
)
def test_invalid_or_duplicate_length_is_not_consumed(headers: list[tuple[str, str]]) -> None:
    body = WireBody((b"unread",))
    result = read_source_body(response_for(body, headers))
    assert result == SourceBodyFailure("LENGTH_REFUSED", 200, 0)
    assert body.reads == 0 and body.closed


@pytest.mark.parametrize("declared", [str(2097153), "9" * 5000])
def test_oversized_declared_length_refused_before_read(declared: str) -> None:
    body = WireBody((b"unread",))
    result = read_source_body(response_for(body, [("Content-Length", declared)]))
    assert result == SourceBodyFailure("BODY_LIMIT", 200, 0)
    assert body.reads == 0 and body.closed


@pytest.mark.parametrize("declared", [None, "1"])
def test_cumulative_limit_without_trusting_declared_size(declared: str | None) -> None:
    body = WireBody((b"x" * MAX_SOURCE_BODY_BYTES, b"y" * 65536, b"unread"))
    headers = [] if declared is None else [("Content-Length", declared)]
    result = read_source_body(response_for(body, headers))
    assert result == SourceBodyFailure("BODY_LIMIT", 200, MAX_SOURCE_BODY_BYTES + 65536)
    assert body.reads == 2 and body.closed
    assert not hasattr(result, "body")


def test_exact_limit_is_accepted_as_transport_only() -> None:
    body = WireBody((b"x" * MAX_SOURCE_BODY_BYTES,))
    result = read_source_body(response_for(body, [("Content-Length", str(MAX_SOURCE_BODY_BYTES))]))
    assert isinstance(result, SourceBody)
    assert len(result.body) == MAX_SOURCE_BODY_BYTES and body.closed


def test_non_ascii_length_is_refused_before_read() -> None:
    body = WireBody((b"unread",))
    response = httpx.Response(200, headers=[(b"Content-Length", b"\xd9\xa1")], stream=body)
    assert read_source_body(response) == SourceBodyFailure("LENGTH_REFUSED", 200, 0)
    assert body.reads == 0 and body.closed


def test_zero_length_with_many_leading_zeros_is_valid() -> None:
    body = WireBody(())
    result = read_source_body(response_for(body, [("Content-Length", "0" * 5000)]))
    assert isinstance(result, SourceBody) and result.body == b"" and body.closed


@pytest.mark.parametrize("declared", ["2", "4"])
def test_completed_length_mismatch_does_not_return_body(declared: str) -> None:
    body = WireBody((b"abc",))
    result = read_source_body(response_for(body, [("Content-Length", declared)]))
    assert result == SourceBodyFailure("LENGTH_MISMATCH", 200, 3)
    assert body.closed


@pytest.mark.parametrize("status", [204, 301, 401, 429, 500])
def test_non_200_body_is_not_consumed_or_retained(status: int) -> None:
    body = WireBody((b"private-error",))
    result = read_source_body(httpx.Response(status, stream=body))
    assert result == SourceBodyFailure("HTTP_ERROR", status, 0)
    assert body.reads == 0 and body.closed


def test_partial_stream_failure_retains_no_payload_or_exception_text() -> None:
    body = WireBody((b"x" * 65536,), fail=True)
    result = read_source_body(response_for(body, []))
    assert result == SourceBodyFailure("TRANSPORT_ERROR", 200, 65536)
    assert "private-account" not in repr(result)
    assert body.closed


def test_preconsumed_response_is_refused_without_existing_content() -> None:
    response = httpx.Response(200, content=b"preconsumed-private-body")
    result = read_source_body(response)
    assert result == SourceBodyFailure("STREAM_REFUSED", 200, 0)
    assert "preconsumed" not in repr(result) and response.is_closed


def test_reader_works_with_httpx_intercepted_stream_without_eager_read() -> None:
    body = WireBody((b'{"transport":"only"}',))

    def intercept(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, stream=body)

    with (
        httpx.Client(transport=httpx.MockTransport(intercept)) as client,
        client.stream("GET", "https://example.invalid/no-market-data") as response,
    ):
        assert body.reads == 0
        result = read_source_body(response)
    assert isinstance(result, SourceBody) and body.closed


def test_partial_unflushed_stream_bytes_are_in_failure_count() -> None:
    body = WireBody((b"partial",), fail=True)
    result = read_source_body(response_for(body, []))
    assert result == SourceBodyFailure("TRANSPORT_ERROR", 200, 7)
    assert body.closed


def test_close_failure_cannot_return_completed_payload() -> None:
    class FailedClose(WireBody):
        def close(self) -> None:
            self.closed = True
            raise httpx.CloseError("private-close-error")

    body = FailedClose((b"transport-only",))
    result = read_source_body(response_for(body, []))
    assert result == SourceBodyFailure("TRANSPORT_ERROR", 200, 14)
    assert body.closed and "private" not in repr(result)


def test_stream_state_error_is_sanitized() -> None:
    class FailedStream(WireBody):
        def __iter__(self) -> Iterator[bytes]:
            yield b"partial"
            raise httpx.StreamClosed()

    body = FailedStream(())
    result = read_source_body(response_for(body, []))
    assert result == SourceBodyFailure("STREAM_REFUSED", 200, 7)
    assert body.closed


def test_deadline_interrupt_is_not_swallowed_even_when_close_fails() -> None:
    class DeadlineStream(WireBody):
        def __iter__(self) -> Iterator[bytes]:
            yield b"partial"
            raise TimeoutError("hard supervisor deadline")

        def close(self) -> None:
            self.closed = True
            raise httpx.CloseError("private-close-error")

    body = DeadlineStream(())
    with pytest.raises(TimeoutError, match="hard supervisor deadline"):
        read_source_body(response_for(body, []))
    assert body.closed


def test_early_refusal_close_error_is_sanitized() -> None:
    class FailedClose(WireBody):
        def close(self) -> None:
            self.closed = True
            raise httpx.CloseError("private-close-error")

    body = FailedClose((b"unread",))
    result = read_source_body(httpx.Response(401, stream=body))
    assert result == SourceBodyFailure("TRANSPORT_ERROR", 401, 0)
    assert body.reads == 0 and body.closed


def test_closed_unconsumed_response_is_refused() -> None:
    body = WireBody((b"unread",))
    response = response_for(body, [])
    response.close()
    result = read_source_body(response)
    assert result == SourceBodyFailure("STREAM_REFUSED", 200, 0)
    assert body.reads == 0 and body.closed


def test_already_read_prefix_is_counted_when_fresh_stream_contract_is_refused() -> None:
    body = WireBody((b"prefix", b"unread"))
    response = response_for(body, [])
    iterator = response.iter_raw()
    assert next(iterator) == b"prefix"
    result = read_source_body(response)
    assert result == SourceBodyFailure("STREAM_REFUSED", 200, 6)
    assert body.reads == 1 and body.closed

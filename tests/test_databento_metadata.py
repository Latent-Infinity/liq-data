import asyncio
import json
from typing import cast
from unittest.mock import Mock

import httpx
import pytest

from liq.data.providers import databento_metadata as metadata


def _body():
    # Availability descriptors only; no market observations.
    return {
        "start": "2010-01-01T00:00:00.000000001Z",
        "end": "2010-01-03T00:00:00.000000009Z",
        "schema": {
            "definition": {
                "start": "2010-01-02T00:00:00.000000002Z",
                "end": "2010-01-03T00:00:00.000000003Z",
            }
        },
    }


def _response(
    payload: object | None = None,
    *,
    body: bytes | None = None,
    content_type: str = "application/json",
    content_encoding: str = "identity",
) -> metadata.RawMetadataResponse:
    content = json.dumps(_body() if payload is None else payload).encode() if body is None else body
    return metadata.RawMetadataResponse(
        metadata.MetadataDataset.MINI, 200, content, content_type, content_encoding
    )


def test_descriptor_preserves_nanos_and_per_schema_ranges() -> None:
    descriptor = metadata.parse_dataset_range(_response())
    assert descriptor.start == "2010-01-01T00:00:00.000000001Z"
    assert descriptor.end == "2010-01-03T00:00:00.000000009Z"
    assert descriptor.schemas[0].start == "2010-01-02T00:00:00.000000002Z"
    assert descriptor.schemas[0].end == "2010-01-03T00:00:00.000000003Z"
    assert descriptor.end_inclusive is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"start": "2010-01-01T00:00:00Z", "end": "2010-01-02T00:00:00Z"},
        {**_body(), "schema": {"definition": {"start": "2010-01-02T00:00:00Z"}}},
        {**_body(), "unrecognized": True},
        {**_body(), "schema": {"x" + str(i): _body()["schema"]["definition"] for i in range(101)}},
    ],
)
def test_incomplete_or_excess_descriptor_refuses(payload) -> None:
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_dataset_range(_response(payload))


@pytest.mark.parametrize(
    "timestamp",
    [
        "2010-01-01T00:00:00",
        "2010-01-01T00:00:00-05:00",
        "2010-02-30T00:00:00Z",
        "2010-01-01T00:00:00.1234567890Z",
        1,
        None,
    ],
)
def test_invalid_utc_timestamp_refuses(timestamp) -> None:
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_dataset_range(_response({**_body(), "start": timestamp}))


@pytest.mark.parametrize(
    "payload",
    [
        {**_body(), "end": _body()["start"]},
        {**_body(), "end": "2009-01-01T00:00:00Z"},
        {
            **_body(),
            "schema": {"definition": {"start": "2009-01-01T00:00:00Z", "end": _body()["end"]}},
        },
    ],
)
def test_invalid_range_refuses_without_dataset_fallback(payload) -> None:
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_dataset_range(_response(payload))


@pytest.mark.parametrize(
    "body", [b'{"start":"first","start":"second"}', b'{"start":NaN}', b"invalid"]
)
def test_duplicate_nonfinite_and_invalid_json_refuse(body: bytes) -> None:
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_dataset_range(_response(body=body))


def test_unknown_dataset_refuses_before_transport() -> None:
    with pytest.raises(metadata.MetadataRefused, match="dataset_scope"):
        metadata.MetadataRequest(cast(metadata.MetadataDataset, "UNKNOWN"))


@pytest.mark.parametrize("dataset", ["EQUS.MINI", "EQUS.SUMMARY"])
def test_metadata_transport_uses_fixed_route_and_one_attempt(dataset: str) -> None:
    witness = Mock()
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            stream=httpx.ByteStream(json.dumps(_body()).encode()),
        )

    response = metadata.fetch_dataset_range(
        metadata.MetadataRequest(metadata.MetadataDataset(dataset)),
        "metadata-only-test-key",
        transport=httpx.MockTransport(handle),
        on_http_attempt=witness,
    )
    assert response.body == json.dumps(_body()).encode()
    assert len(requests) == 1
    assert (
        str(requests[0].url)
        == "https://hist.databento.com/v0/metadata.get_dataset_range?dataset=" + dataset
    )
    assert requests[0].method == "GET"
    assert requests[0].headers["accept-encoding"] == "identity"
    assert requests[0].headers["authorization"].startswith("Basic ")
    witness.assert_called_once_with()
    assert "metadata-only-test-key" not in repr(response)


@pytest.mark.parametrize("status", [302, 403, 429, 500])
def test_error_response_is_retained_before_refusal_without_retry(status: int) -> None:
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://other.invalid"},
            stream=httpx.ByteStream(b"nonfinancial error body"),
        )

    response = metadata.fetch_dataset_range(
        metadata.MetadataRequest(metadata.MetadataDataset.MINI),
        "metadata-only-test-key",
        transport=httpx.MockTransport(handle),
        on_http_attempt=lambda: None,
    )
    assert response.http_status == status
    assert response.body == b"nonfinancial error body"
    assert len(requests) == 1
    with pytest.raises(metadata.MetadataRefused, match="http_refused"):
        metadata.parse_dataset_range(response)


def test_body_limit_refuses_and_closes_stream() -> None:
    stream = httpx.ByteStream(b"x" * 65537)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    with pytest.raises(metadata.MetadataRefused, match="response_byte_limit"):
        metadata.fetch_dataset_range(
            metadata.MetadataRequest(metadata.MetadataDataset.MINI),
            "metadata-only-test-key",
            transport=transport,
            on_http_attempt=lambda: None,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"content_encoding": "gzip"},
        {"content_type": "text/html"},
    ],
)
def test_unexpected_encoding_or_type_refuses(changes) -> None:
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_dataset_range(_response(**changes))


def test_missing_credentials_precede_http_attempt() -> None:
    witness = Mock()
    with pytest.raises(metadata.MetadataRefused, match="missing_credentials"):
        metadata.fetch_dataset_range(
            metadata.MetadataRequest(metadata.MetadataDataset.MINI), "", on_http_attempt=witness
        )
    witness.assert_not_called()


def test_full_request_deadline_cancels_and_closes_transport(monkeypatch) -> None:
    class Slow(httpx.AsyncBaseTransport):
        closed = False

        async def handle_async_request(self, request):
            await asyncio.sleep(60)
            return httpx.Response(200)

        async def aclose(self):
            self.closed = True

    transport = Slow()
    witness = Mock()
    monkeypatch.setattr(metadata, "_REQUEST_SECONDS", 0.02)
    with pytest.raises(metadata.MetadataRefused, match="request_deadline"):
        metadata.fetch_dataset_range(
            metadata.MetadataRequest(metadata.MetadataDataset.MINI),
            "metadata-only-test-key",
            transport=transport,
            on_http_attempt=witness,
        )
    witness.assert_called_once_with()
    assert transport.closed


def test_transport_error_has_safe_reason_and_one_attempt() -> None:
    def handle(request):
        raise httpx.ReadError("untrusted transport error", request=request)

    witness = Mock()
    with pytest.raises(metadata.MetadataRefused, match="transport_error") as captured:
        metadata.fetch_dataset_range(
            metadata.MetadataRequest(metadata.MetadataDataset.MINI),
            "metadata-only-test-key",
            transport=httpx.MockTransport(handle),
            on_http_attempt=witness,
        )
    assert "untrusted transport error" not in str(captured.value)
    witness.assert_called_once_with()


@pytest.mark.parametrize(
    "headers,body",
    [
        ({"Content-Encoding": "gzip"}, b"opaque"),
        ({"Content-Length": "65537"}, b"opaque"),
        ({"Content-Length": "invalid"}, b"opaque"),
        ({"Content-Length": "10"}, b"short"),
    ],
)
def test_invalid_transport_envelope_refuses(headers, body) -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, headers=headers, stream=httpx.ByteStream(body))
    )
    with pytest.raises(metadata.MetadataRefused):
        metadata.fetch_dataset_range(
            metadata.MetadataRequest(metadata.MetadataDataset.MINI),
            "metadata-only-test-key",
            transport=transport,
            on_http_attempt=lambda: None,
        )


def test_sync_client_refuses_active_event_loop_before_attempt() -> None:
    witness = Mock()

    async def invoke():
        with pytest.raises(metadata.MetadataRefused, match="requires_sync_context"):
            metadata.fetch_dataset_range(
                metadata.MetadataRequest(metadata.MetadataDataset.MINI),
                "metadata-only-test-key",
                on_http_attempt=witness,
            )

    asyncio.run(invoke())
    witness.assert_not_called()


def test_empty_schema_does_not_inherit_dataset_range() -> None:
    descriptor = metadata.parse_dataset_range(_response({**_body(), "schema": {}}))
    assert descriptor.schemas == ()


def test_nasdaq_override_has_exact_fixed_metadata_scope() -> None:
    entries = []

    def respond(request):
        entries.append((request.url.path, dict(request.url.params)))
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            stream=httpx.ByteStream(json.dumps(_body()).encode()),
        )

    response = metadata.fetch_dataset_range(
        metadata.MetadataRequest(metadata.MetadataDataset.NASDAQ),
        "metadata-only-test-key",
        transport=httpx.MockTransport(respond),
        on_http_attempt=lambda: None,
    )
    assert entries == [("/v0/metadata.get_dataset_range", {"dataset": "XNAS.ITCH"})]
    assert metadata.parse_dataset_range(response).dataset is metadata.MetadataDataset.NASDAQ

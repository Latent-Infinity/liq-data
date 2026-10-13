from decimal import Decimal
from typing import cast
from unittest.mock import Mock

import httpx
import pytest

from liq.data.providers import databento_metadata as metadata


def _request() -> metadata.DefinitionCostRequest:
    return metadata.DefinitionCostRequest(
        start="2018-05-01T00:00:00Z", end="2018-05-02T00:00:00Z", limit=1000
    )


def _response(body: bytes, *, status: int = 200) -> metadata.RawMetadataResponse:
    return metadata.RawMetadataResponse(
        metadata.MetadataDataset.NASDAQ, status, body, "application/json", "identity"
    )


def test_cost_query_is_one_documented_get_with_exact_definition_scope() -> None:
    witness = Mock()
    observed = []

    def respond(request):
        observed.append(
            (request.method, str(request.url.copy_with(query=None)), dict(request.url.params))
        )
        return httpx.Response(
            200, headers={"Content-Type": "application/json"}, stream=httpx.ByteStream(b"0")
        )

    response = metadata.fetch_definition_cost(
        _request(),
        "metadata-only-test-key",
        on_http_attempt=witness,
        transport=httpx.MockTransport(respond),
    )
    assert observed == [
        (
            "GET",
            "https://hist.databento.com/v0/metadata.get_cost",
            {
                "dataset": "XNAS.ITCH",
                "schema": "definition",
                "symbols": "ALL_SYMBOLS",
                "stype_in": "raw_symbol",
                "start": "2018-05-01T00:00:00Z",
                "end": "2018-05-02T00:00:00Z",
                "limit": "1000",
            },
        )
    ]
    witness.assert_called_once_with()
    assert response.body == b"0"
    estimate = metadata.parse_definition_cost(response, _request())
    assert estimate.estimated_usd == Decimal(0)
    assert estimate.complete_universe is False
    assert estimate.record_limit == 1000


@pytest.mark.parametrize(
    "body", [b"0", b"0.000000000000000001", b"1e-9999", b"1.234567890123456789"]
)
def test_exact_cost_never_rounds_positive_values_to_zero(body: bytes) -> None:
    estimate = metadata.parse_definition_cost(_response(body), _request())
    assert estimate.estimated_usd == Decimal(body.decode())
    if body != b"0":
        assert estimate.estimated_usd > 0
    document = estimate.model_dump(mode="json")
    assert Decimal(document["estimated_usd"]) == estimate.estimated_usd


@pytest.mark.parametrize(
    "body",
    [
        b"true",
        b"false",
        b'"0"',
        b"[]",
        b"{}",
        b"null",
        b"-1",
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"1 2",
        b"",
        b"1e999999999999999999999999",
    ],
)
def test_cost_shapes_and_nonfinite_values_are_refused(body: bytes) -> None:
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_definition_cost(_response(body), _request())


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": metadata.MetadataDataset.MINI},
        {"limit": 0},
        {"limit": True},
        {"limit": 1001},
        {"start": "2018-05-01T01:00:00Z"},
        {"start": "2018-05-01"},
        {"end": "2018-05-03T00:00:00Z"},
        {"end": "2018-05-01T00:00:00Z"},
    ],
)
def test_definition_cost_request_refuses_unsupported_scope(changes) -> None:
    values = {
        "start": "2018-05-01T00:00:00Z",
        "end": "2018-05-02T00:00:00Z",
        "limit": 1000,
        "dataset": metadata.MetadataDataset.NASDAQ,
        **changes,
    }
    with pytest.raises(metadata.MetadataRefused):
        metadata.DefinitionCostRequest(
            start=cast(str, values["start"]),
            end=cast(str, values["end"]),
            limit=cast(int, values["limit"]),
            dataset=cast(metadata.MetadataDataset, values["dataset"]),
        )


def test_refused_get_never_falls_back_to_sdk_post_or_redirect() -> None:
    methods = []
    witness = Mock()

    def respond(request):
        methods.append(request.method)
        return httpx.Response(
            302,
            headers={"Location": "https://example.invalid"},
            stream=httpx.ByteStream(b"opaque refusal"),
        )

    raw = metadata.fetch_definition_cost(
        _request(),
        "metadata-only-test-key",
        on_http_attempt=witness,
        transport=httpx.MockTransport(respond),
    )
    assert methods == ["GET"]
    assert raw.body == b"opaque refusal"
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_definition_cost(raw, _request())
    witness.assert_called_once_with()


def test_other_dataset_cost_response_is_refused() -> None:
    response = metadata.RawMetadataResponse(
        metadata.MetadataDataset.MINI, 200, b"0", "application/json", "identity"
    )
    with pytest.raises(metadata.MetadataRefused):
        metadata.parse_definition_cost(response, _request())


def test_range_request_cannot_dispatch_cost_method() -> None:
    witness = Mock()
    with pytest.raises(metadata.MetadataRefused):
        metadata.fetch_definition_cost(
            cast(
                metadata.DefinitionCostRequest,
                metadata.MetadataRequest(metadata.MetadataDataset.NASDAQ),
            ),
            "metadata-only-test-key",
            on_http_attempt=witness,
        )
    witness.assert_not_called()

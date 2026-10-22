import json
from datetime import date
from urllib.parse import parse_qs

import httpx
import pytest
from pydantic import ValidationError

from liq.data.alpaca_action_schema import (
    AlpacaActionSchemaRequest,
    guard_action_schema,
    validate_action_schema_request,
)
from liq.data.alpaca_source import AlpacaCaptureRefused, AlpacaCorporateActionsRequest
from liq.data.providers.alpaca import AlpacaProvider


@pytest.mark.parametrize(
    "change",
    [
        {"symbols": ("SPY", "KHC")},
        {"start": date(2022, 1, 2)},
        {"end": date(2023, 4, 1)},
        {"data_quality": "all"},
        {"max_pages": 2},
        {"limit": 1001},
        {"max_response_bytes": 1048577},
    ],
)
def test_request_refuses_scope_changes(change):
    with pytest.raises(ValidationError):
        AlpacaActionSchemaRequest.model_validate(change)


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "characterization"},
        {"arm_id": "other"},
        {"final_portfolio_review": True},
    ],
)
def test_guard_refuses_declaration_changes_before_logging(tmp_path, change):
    log = tmp_path / "usage.jsonl"
    with pytest.raises(AlpacaCaptureRefused):
        guard_action_schema(AlpacaActionSchemaRequest(), log, **change)
    assert not log.exists()


def test_guard_records_exact_new_scope_without_changing_old_scope(tmp_path):
    log = tmp_path / "usage.jsonl"
    guard_action_schema(AlpacaActionSchemaRequest(), log)
    entry = json.loads(log.read_text())
    assert entry["purpose"] == "dev_smoke"
    assert entry["arm_id"] == "classical_cross_sectional_ranker"
    assert entry["final_portfolio_review"] is False
    with pytest.raises(ValidationError):
        AlpacaCorporateActionsRequest(
            symbols=("KHC", "SPY"), start=date(2022, 1, 1), end=date(2023, 3, 31)
        )


def test_provider_uses_one_exact_request_and_never_follows_cursor():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b'{"corporate_actions":{},"next_page_token":"opaque"}'),
        )

    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(respond)
    raw = provider.fetch_source_page(AlpacaActionSchemaRequest(), page_token=None)
    assert len(calls) == 1
    assert parse_qs(calls[0].url.query.decode()) == {
        "symbols": ["KHC,SPY"],
        "start": ["2022-01-01"],
        "end": ["2023-03-31"],
        "data_quality": ["complete"],
        "limit": ["1000"],
        "sort": ["asc"],
        "region": ["us"],
    }
    assert b'"opaque"' in raw
    with pytest.raises(AlpacaCaptureRefused, match="continuation_forbidden"):
        provider.fetch_source_page(AlpacaActionSchemaRequest(), page_token="opaque")
    assert len(calls) == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_provider_refuses_http_status_without_retry(status):
    attempts = []
    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(lambda _request: httpx.Response(status))
    with pytest.raises(AlpacaCaptureRefused) as refused:
        provider.fetch_source_page(
            AlpacaActionSchemaRequest(), page_token=None, on_http_attempt=lambda: attempts.append(1)
        )
    assert refused.value.status_code == status
    assert len(attempts) == 1


def test_provider_refuses_forged_scope_before_http():
    attempts = []
    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(lambda _request: httpx.Response(200))
    forged = AlpacaActionSchemaRequest.model_construct(symbols=("AAPL", "SPY"))
    with pytest.raises(AlpacaCaptureRefused):
        provider.fetch_source_page(
            forged, page_token=None, on_http_attempt=lambda: attempts.append(1)
        )
    assert attempts == []


def test_guard_refuses_subclass_without_calling_its_methods(tmp_path):
    calls = []

    class Subclass(AlpacaActionSchemaRequest):
        def model_dump(self, **kwargs):
            calls.append(1)
            return super().model_dump(**kwargs)

    with pytest.raises(AlpacaCaptureRefused, match="request_type"):
        guard_action_schema(Subclass(), tmp_path / "usage.jsonl")
    assert calls == []


def test_validator_refuses_wrong_type_before_attribute_access():
    with pytest.raises(AlpacaCaptureRefused, match="request_type"):
        validate_action_schema_request(None)

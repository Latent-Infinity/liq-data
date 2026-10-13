import asyncio
import time
from datetime import UTC, date, datetime
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic import ValidationError

from liq.data.alpaca_source import (
    AlpacaBarsRequest,
    AlpacaCaptureRefused,
    AlpacaCorporateActionsRequest,
)
from liq.data.exceptions import LockboxViolationError, UndeclaredReadError
from liq.data.lockbox import CLASSICAL_RANKER_ALPACA_LEDGER_V1, resolve_dataset
from liq.data.providers import alpaca as alpaca_module
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.service import DataService


class AsyncByteBody(httpx.AsyncByteStream):
    def __init__(self, body: bytes):
        self.body = body

    async def __aiter__(self):
        yield self.body

    async def aclose(self):
        pass


def test_source_page_enforces_total_request_deadline(monkeypatch):
    monkeypatch.setattr(alpaca_module, "SOURCE_PAGE_TIMEOUT_SECONDS", 0.02)

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"bars":[]'
            await asyncio.sleep(0.12)
            yield b',"next_page_token":null}'

        async def aclose(self):
            pass

    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={
                "content-encoding": "identity",
                "content-type": "application/json",
            },
            stream=SlowStream(),
        )
    )
    request = AlpacaBarsRequest(
        symbol="KHC",
        start=datetime(2023, 4, 21, 13, 30, tzinfo=UTC),
        end=datetime(2023, 4, 24, 19, 59, tzinfo=UTC),
        asof=date(2023, 4, 24),
        adjustment="raw",
        feed="sip",
    )

    started = time.monotonic()
    with pytest.raises(AlpacaCaptureRefused, match="request_deadline"):
        provider.fetch_source_page(request, page_token=None)
    assert time.monotonic() - started < 0.1


def test_bars_request_binds_adjustment_entity_feed_and_utc_dates():
    request = AlpacaBarsRequest(
        symbol="KHC",
        start=datetime(2023, 4, 21, 13, 30, tzinfo=UTC),
        end=datetime(2023, 4, 24, 19, 59, tzinfo=UTC),
        asof=date(2023, 4, 24),
        adjustment="raw",
        feed="sip",
    )
    assert request.symbol == "KHC"
    assert request.adjustment == "raw"
    assert request.feed == "sip"
    assert request.asof == date(2023, 4, 24)


def test_bar_basis_and_symbol_identity_are_required():
    with pytest.raises(ValidationError):
        AlpacaBarsRequest.model_validate(
            {
                "symbol": "KHC",
                "start": datetime(2023, 4, 21, tzinfo=UTC),
                "end": datetime(2023, 4, 24, tzinfo=UTC),
                "adjustment": "raw",
                "feed": "sip",
            }
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"adjustment": "all,raw"},
        {"feed": "unknown"},
        {"symbol": "AAPL"},
        {"start": datetime(2023, 4, 21)},
        {"start": datetime(2023, 4, 22, tzinfo=UTC)},
        {"max_pages": 0},
    ],
)
def test_invalid_bar_capture_request_refuses(overrides):
    values = {
        "symbol": "KHC",
        "start": datetime(2023, 4, 21, 13, 30, tzinfo=UTC),
        "end": datetime(2023, 4, 24, 19, 59, tzinfo=UTC),
        "asof": date(2023, 4, 24),
        "adjustment": "raw",
        "feed": "sip",
    }
    values.update(overrides)
    with pytest.raises(ValidationError):
        AlpacaBarsRequest.model_validate(values)


def test_action_request_uses_provider_process_date_bounds():
    request = AlpacaCorporateActionsRequest(
        symbols=("LEH", "LEHMQ"), start=date(2008, 1, 1), end=date(2008, 12, 31)
    )
    assert request.start == date(2008, 1, 1)
    assert request.end == date(2008, 12, 31)
    assert request.data_quality == "complete"


def test_action_request_refuses_wrong_window_universe_and_quality():
    with pytest.raises(ValidationError):
        AlpacaCorporateActionsRequest(
            symbols=("KHC", "SPY"), start=date(2007, 1, 1), end=date(2008, 12, 31)
        )
    with pytest.raises(ValidationError):
        AlpacaCorporateActionsRequest(
            symbols=("LEH", "LEHMQ"), start=date(2008, 1, 1), end=date(2009, 12, 31)
        )
    with pytest.raises(ValidationError):
        AlpacaCorporateActionsRequest.model_validate(
            {
                "symbols": ("LEH", "LEHMQ"),
                "start": date(2008, 1, 1),
                "end": date(2008, 12, 31),
                "data_quality": "all",
            }
        )


def test_resolver_has_dedicated_alpaca_datasets():
    assert resolve_dataset("alpaca", "KHC", data_kind="ranker_bars") == "alpaca_ranker_bars"
    assert (
        resolve_dataset("alpaca", "LEH", data_kind="ranker_corporate_actions")
        == "alpaca_ranker_corporate_actions"
    )
    assert resolve_dataset("alpaca", "KHC") is None
    assert resolve_dataset("databento", "KHC", data_kind="ranker_bars") is None


@pytest.fixture
def service(tmp_path):
    return DataService(data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ALPACA_LEDGER_V1)


def action_request():
    return AlpacaCorporateActionsRequest(
        symbols=("LEH", "LEHMQ"), start=date(2008, 1, 1), end=date(2008, 12, 31)
    )


def test_alpaca_capture_guard_precedes_provider_and_binds_page(service):
    provider = AlpacaProvider(api_key="test", api_secret="test")
    with (
        patch.object(
            provider, "fetch_source_page", return_value=b'{"corporate_actions":{}}'
        ) as page,
        patch.object(service, "_get_provider", return_value=provider),
    ):
        response = service.fetch_alpaca_source_page(
            action_request(),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    assert response == b'{"corporate_actions":{}}'
    page.assert_called_once_with(action_request(), page_token=None)


@pytest.mark.parametrize(
    "purpose,arm_id",
    [("discovery", "classical_cross_sectional_ranker"), ("characterization", "other")],
)
def test_alpaca_capture_wrong_scope_refuses_before_provider(service, purpose, arm_id):
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            action_request(), page_token=None, purpose=purpose, arm_id=arm_id
        )
    provider.assert_not_called()


def test_alpaca_capture_requires_declared_purpose_before_provider(service):
    with patch.object(service, "_get_provider") as provider, pytest.raises(UndeclaredReadError):
        service.fetch_alpaca_source_page(
            action_request(),
            page_token=None,
            purpose=None,
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


def test_alpaca_capture_requires_exact_dedicated_ledger(tmp_path):
    service = DataService(data_root=tmp_path)
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            action_request(),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


def test_corporate_action_page_binds_query_and_returns_body():
    from liq.data.providers.alpaca import AlpacaProvider

    payload = b'{"corporate_actions":{"cash_dividends":[]},"next_page_token":null}'
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-encoding": "identity", "content-type": "application/json"},
            stream=AsyncByteBody(payload),
        )

    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(handler)
    request = AlpacaCorporateActionsRequest(
        symbols=("LEH", "LEHMQ"), start=date(2008, 1, 1), end=date(2008, 12, 31)
    )
    assert provider.fetch_source_page(request, page_token=None) == payload
    query = parse_qs(urlparse(str(seen[0].url)).query)
    assert query["symbols"] == ["LEH,LEHMQ"]
    assert query["start"] == ["2008-01-01"]
    assert query["end"] == ["2008-12-31"]
    assert query["data_quality"] == ["complete"]
    assert query["region"] == ["us"]


@pytest.mark.parametrize(
    ("status", "encoding"),
    [(302, "identity"), (403, "identity"), (200, "gzip")],
)
def test_corporate_action_page_refuses_redirect_error_and_encoding(status, encoding):
    from liq.data.alpaca_source import AlpacaCaptureRefused
    from liq.data.providers.alpaca import AlpacaProvider

    def handler(_request):
        return httpx.Response(
            status,
            headers={"content-encoding": encoding},
            stream=AsyncByteBody(b"private"),
        )

    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(handler)
    request = AlpacaCorporateActionsRequest(
        symbols=("LEH", "LEHMQ"), start=date(2008, 1, 1), end=date(2008, 12, 31)
    )
    with pytest.raises(AlpacaCaptureRefused):
        provider.fetch_source_page(request, page_token=None)


def test_source_page_refuses_over_budget_body_before_returning_a_prefix():
    from liq.data.alpaca_source import AlpacaCaptureRefused

    body = b'{"corporate_actions":{"note":"' + b"x" * 32 + b'"}}'
    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-encoding": "identity", "content-type": "application/json"},
            stream=AsyncByteBody(body),
        )
    )
    request = AlpacaCorporateActionsRequest(
        symbols=("LEH", "LEHMQ"),
        start=date(2008, 1, 1),
        end=date(2008, 12, 31),
    ).model_copy(update={"max_response_bytes": 16})
    with pytest.raises(AlpacaCaptureRefused) as refused:
        provider.fetch_source_page(request, page_token=None)
    assert str(refused.value) == "response_byte_limit"


def test_bars_page_binds_basis_asof_feed_and_window():
    payload = b'{"bars":[],"next_page_token":null}'
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-encoding": "identity", "content-type": "application/json"},
            stream=AsyncByteBody(payload),
        )

    provider = AlpacaProvider(api_key="test", api_secret="test")
    provider._source_transport = httpx.MockTransport(handler)
    request = AlpacaBarsRequest(
        symbol="KHC",
        start=datetime(2023, 4, 21, 13, 30, tzinfo=UTC),
        end=datetime(2023, 4, 24, 19, 59, tzinfo=UTC),
        asof=date(2023, 4, 24),
        adjustment="all",
        feed="sip",
    )
    assert provider.fetch_source_page(request, page_token=None) == payload
    query = parse_qs(urlparse(str(seen[0].url)).query)
    assert query["symbols"] == ["KHC"]
    assert query["start"] == ["2023-04-21T13:30:00Z"]
    assert query["end"] == ["2023-04-24T19:59:00Z"]
    assert query["adjustment"] == ["all"]
    assert query["asof"] == ["2023-04-24"]
    assert query["feed"] == ["sip"]
    assert query["currency"] == ["USD"]
    assert query["timeframe"] == ["1Min"]


def test_bars_page_preserves_basis_as_cache_separate_request_identity():
    raw = AlpacaBarsRequest(
        symbol="SPY",
        start=datetime(2023, 4, 21, 13, 30, tzinfo=UTC),
        end=datetime(2023, 4, 24, 19, 59, tzinfo=UTC),
        asof=date(2023, 4, 24),
        adjustment="raw",
        feed="sip",
    )
    adjusted = raw.model_copy(update={"adjustment": "all"})
    assert raw.adjustment != adjusted.adjustment
    assert raw != adjusted

"""Source-free controls for exact KHC requests, guards and bounded transport."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from unittest.mock import patch

import anyio
import httpx
import pytest
from pydantic import ValidationError

from liq.data.alpaca_momentum_source import (
    AlpacaMomentumCalendarRequest,
    AlpacaMomentumDailyRequest,
)
from liq.data.alpaca_source import (
    AlpacaCalendarRequest,
    AlpacaCaptureRefused,
    AlpacaDailyFixtureRequest,
)
from liq.data.exceptions import LockboxViolationError
from liq.data.lockbox import CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1, resolve_dataset
from liq.data.providers import alpaca as alpaca_module
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.service import DataService


class EmptyJSON(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"[]"

    async def aclose(self) -> None:
        return None


@pytest.mark.parametrize(
    "request_type", [AlpacaMomentumCalendarRequest, AlpacaMomentumDailyRequest]
)
def test_request_scope_is_immutable_and_separate(request_type) -> None:
    request = request_type()
    assert request.max_pages == 1
    assert request.max_response_bytes == 131072
    with pytest.raises(ValidationError):
        request.max_pages = 2
    with pytest.raises(ValidationError):
        request_type(max_response_bytes=131073)
    with pytest.raises(ValidationError):
        request_type(end=date(2023, 4, 1))


@pytest.mark.parametrize(
    "changed",
    [
        {"symbol": "SPY"},
        {"start": datetime(2022, 4, 2, 4, tzinfo=UTC)},
        {"end": datetime(2023, 4, 1, 4, tzinfo=UTC)},
        {"asof": date(2023, 4, 1)},
        {"adjustment": "raw"},
        {"feed": "iex"},
        {"limit": 1000},
    ],
)
def test_daily_scope_refuses_variants(changed) -> None:
    with pytest.raises(ValidationError):
        AlpacaMomentumDailyRequest(**changed)


@pytest.mark.parametrize(
    "request_type,kind,dataset",
    [
        (
            AlpacaMomentumCalendarRequest,
            "ranker_momentum_calendar_reference",
            "alpaca_ranker_momentum_calendar_reference",
        ),
        (
            AlpacaMomentumDailyRequest,
            "ranker_momentum_daily_fixture",
            "alpaca_ranker_momentum_daily_fixture",
        ),
    ],
)
def test_exact_datasets_are_logged_before_provider(tmp_path, request_type, kind, dataset) -> None:
    assert resolve_dataset("alpaca", "KHC", data_kind=kind) == dataset
    assert resolve_dataset("databento", "KHC", data_kind=kind) is None
    service = DataService(
        data_root=tmp_path,
        research_log_root=tmp_path,
        lockbox_ledger=CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1,
    )
    source = AlpacaProvider(api_key="test", api_secret="test")
    with (
        patch.object(service, "_get_provider", return_value=source),
        patch.object(source, "fetch_source_page", return_value=b"[]"),
    ):
        assert (
            service.fetch_alpaca_source_page(
                request_type(),
                page_token=None,
                purpose="characterization",
                arm_id="classical_cross_sectional_ranker",
            )
            == b"[]"
        )
    logs = list(tmp_path.rglob("*usage*.jsonl"))
    assert len(logs) == 1
    entry = json.loads(logs[0].read_text().splitlines()[0])
    assert entry["dataset"] == dataset


@pytest.mark.parametrize(
    "request_type", [AlpacaMomentumCalendarRequest, AlpacaMomentumDailyRequest]
)
@pytest.mark.parametrize(
    "purpose,arm_id,page_token,final_review",
    [
        ("dev_smoke", "classical_cross_sectional_ranker", None, False),
        ("characterization", "other", None, False),
        ("characterization", "classical_cross_sectional_ranker", "extra", False),
        ("characterization", "classical_cross_sectional_ranker", None, True),
    ],
)
def test_guard_rejects_changed_authority_before_provider(
    tmp_path, request_type, purpose: str, arm_id: str, page_token: str | None, final_review: bool
) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            request_type(),
            page_token=page_token,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_review,
        )
    provider.assert_not_called()


@pytest.mark.parametrize(
    "request_type", [AlpacaMomentumCalendarRequest, AlpacaMomentumDailyRequest]
)
def test_default_ledger_and_unvalidated_date_refuse_before_provider(tmp_path, request_type) -> None:
    service = DataService(data_root=tmp_path)
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            request_type(),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1
    )
    request = request_type().model_copy(
        update={"symbol": "SPY"}
        if request_type is AlpacaMomentumDailyRequest
        else {"start": date(2022, 5, 1)}
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(ValidationError):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


@pytest.mark.parametrize(
    "request_type,path",
    [
        (AlpacaMomentumCalendarRequest, "/v2/calendar"),
        (AlpacaMomentumDailyRequest, "/v2/stocks/bars"),
    ],
)
def test_transport_uses_exact_new_scope_and_one_attempt(request_type, path) -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=EmptyJSON())

    source = AlpacaProvider(api_key="test", api_secret="test")
    source._source_transport = httpx.MockTransport(respond)
    attempts: list[bool] = []
    assert (
        source.fetch_source_page(
            request_type(), page_token=None, on_http_attempt=lambda: attempts.append(True)
        )
        == b"[]"
    )
    assert len(seen) == len(attempts) == 1
    assert seen[0].url.path == path
    if request_type is AlpacaMomentumDailyRequest:
        assert seen[0].url.params["symbols"] == "KHC"
        assert seen[0].url.params["limit"] == "400"
    with pytest.raises(AlpacaCaptureRefused):
        source.fetch_source_page(request_type(), page_token="extra")
    assert len(seen) == 1


def test_old_request_scopes_remain_unchanged() -> None:
    assert AlpacaCalendarRequest().start == date(2023, 1, 26)
    assert AlpacaDailyFixtureRequest(symbol="KHC").limit == 1000


@pytest.mark.parametrize(
    "request_type", [AlpacaMomentumCalendarRequest, AlpacaMomentumDailyRequest]
)
@pytest.mark.parametrize("status", [302, 401, 429, 503])
def test_refusal_does_not_retry_or_follow_redirect(request_type, status: int) -> None:
    attempts: list[bool] = []
    source = AlpacaProvider(api_key="test", api_secret="test")
    source._source_transport = httpx.MockTransport(
        lambda _: httpx.Response(status, headers={"location": "https://example.invalid"})
    )
    with pytest.raises(AlpacaCaptureRefused, match="http_refused"):
        source.fetch_source_page(
            request_type(), page_token=None, on_http_attempt=lambda: attempts.append(True)
        )
    assert attempts == [True]


@pytest.mark.parametrize(
    "request_type", [AlpacaMomentumCalendarRequest, AlpacaMomentumDailyRequest]
)
def test_complete_http_deadline_cancels_single_attempt(monkeypatch, request_type) -> None:
    async def stalled(_: httpx.Request) -> httpx.Response:
        await anyio.sleep(1)
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b"[]")

    monkeypatch.setattr(alpaca_module, "SOURCE_PAGE_TIMEOUT_SECONDS", 0.01)
    source = AlpacaProvider(api_key="test", api_secret="test")
    source._source_transport = httpx.MockTransport(stalled)
    attempts: list[bool] = []
    with pytest.raises(AlpacaCaptureRefused, match="request_deadline"):
        source.fetch_source_page(
            request_type(), page_token=None, on_http_attempt=lambda: attempts.append(True)
        )
    assert attempts == [True]

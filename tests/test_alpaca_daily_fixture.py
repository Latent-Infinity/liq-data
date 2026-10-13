"""Nonfinancial routing controls for the independently scoped daily fixture."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from types import MappingProxyType
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from liq.data.alpaca_source import (
    AlpacaBarsRequest,
    AlpacaCaptureRefused,
    AlpacaCorporateActionsRequest,
    AlpacaDailyFixtureRequest,
)
from liq.data.exceptions import LockboxViolationError
from liq.data.lockbox import (
    CLASSICAL_RANKER_ALPACA_LEDGER_V1,
    CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1,
    FoldWindows,
    resolve_dataset,
)
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.service import DataService


def test_daily_request_has_exact_inclusive_dst_label_bounds() -> None:
    request = AlpacaDailyFixtureRequest(symbol="KHC")
    assert request.start == datetime(2023, 1, 26, 5, tzinfo=UTC)
    assert request.end == datetime(2023, 4, 24, 4, tzinfo=UTC)
    assert request.asof == date(2023, 4, 24)
    assert (request.timeframe, request.adjustment, request.feed) == ("1Day", "all", "sip")
    assert request.max_pages == 1


def test_daily_request_normalizes_explicit_offset_bounds_to_utc() -> None:
    request = AlpacaDailyFixtureRequest(
        symbol="KHC",
        start=datetime(2023, 1, 26, tzinfo=timezone(timedelta(hours=-5))),
        end=datetime(2023, 4, 24, tzinfo=timezone(timedelta(hours=-4))),
    )
    assert request.start.tzinfo == UTC
    assert request.end.tzinfo == UTC


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "AAPL"},
        {"timeframe": "1Min"},
        {"adjustment": "raw"},
        {"feed": "iex"},
        {"currency": "EUR"},
        {"max_pages": 2},
        {"limit": 999},
        {"max_response_bytes": 2097152},
        {"asof": date(2023, 4, 25)},
        {"start": datetime(2023, 1, 26, 4, tzinfo=UTC)},
        {"end": datetime(2023, 4, 24, 5, tzinfo=UTC)},
        {"start": datetime(2023, 1, 26, 5)},
    ],
)
def test_daily_request_refuses_changed_scope(change) -> None:
    with pytest.raises(ValidationError):
        AlpacaDailyFixtureRequest.model_validate({"symbol": "KHC", **change})


def test_daily_dataset_requires_explicit_kind() -> None:
    assert (
        resolve_dataset("alpaca", "KHC", data_kind="ranker_daily_fixture")
        == "alpaca_ranker_daily_fixture"
    )
    assert resolve_dataset("alpaca", "KHC") is None
    assert resolve_dataset("databento", "KHC", data_kind="ranker_daily_fixture") is None


@pytest.mark.parametrize("old_ledger", [None, CLASSICAL_RANKER_ALPACA_LEDGER_V1])
def test_daily_request_refuses_old_or_default_ledger_before_provider(tmp_path, old_ledger) -> None:
    kwargs = {} if old_ledger is None else {"lockbox_ledger": old_ledger}
    service = DataService(data_root=tmp_path, **kwargs)
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaDailyFixtureRequest(symbol="KHC"),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


def test_daily_request_refuses_custom_broader_ledger_before_provider(tmp_path) -> None:
    ledger = replace(
        CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1,
        datasets=MappingProxyType(
            {
                "alpaca_ranker_daily_fixture": FoldWindows(
                    characterization=(date(2023, 1, 1), date(2023, 12, 31)),
                    allowed_purposes=frozenset({"characterization"}),
                ),
            }
        ),
    )
    service = DataService(data_root=tmp_path, lockbox_ledger=ledger)
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaDailyFixtureRequest(symbol="KHC"),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


@pytest.mark.parametrize(
    "purpose,arm,final",
    [
        ("discovery", "classical_cross_sectional_ranker", False),
        ("characterization", "other", False),
        ("characterization", "classical_cross_sectional_ranker", True),
    ],
)
def test_daily_wrong_use_refuses_before_provider(tmp_path, purpose, arm, final) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaDailyFixtureRequest(symbol="KHC"),
            page_token=None,
            purpose=purpose,
            arm_id=arm,
            final_portfolio_review=final,
        )
    provider.assert_not_called()


def test_daily_continuation_refuses_before_provider(tmp_path) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaDailyFixtureRequest(symbol="KHC"),
            page_token="next",
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


def test_minute_request_cannot_use_daily_ledger(tmp_path) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1
    )
    request = AlpacaBarsRequest(
        symbol="KHC",
        start=datetime(2023, 4, 21, 13, 30, tzinfo=UTC),
        end=datetime(2023, 4, 24, 19, 59, tzinfo=UTC),
        asof=date(2023, 4, 24),
        adjustment="raw",
        feed="sip",
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


def test_action_request_cannot_use_daily_ledger(tmp_path) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1
    )
    request = AlpacaCorporateActionsRequest(
        symbols=("LEH", "LEHMQ"), start=date(2008, 1, 1), end=date(2008, 12, 31)
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


def test_provider_refuses_daily_continuation_before_http() -> None:
    source = AlpacaProvider(api_key="test", api_secret="test")
    requests: list[httpx.Request] = []

    def unexpected(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("daily continuation reached HTTP")

    source._source_transport = httpx.MockTransport(unexpected)
    with pytest.raises(AlpacaCaptureRefused, match="continuation_forbidden"):
        source.fetch_source_page(AlpacaDailyFixtureRequest(symbol="KHC"), page_token="next")
    assert requests == []


def test_daily_guard_routes_exact_request_and_logs_dataset(tmp_path) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_DAILY_FIXTURE_LEDGER_V1
    )
    request = AlpacaDailyFixtureRequest(symbol="SPY")
    source = AlpacaProvider(api_key="test", api_secret="test")
    with (
        patch.object(service, "_get_provider", return_value=source),
        patch.object(source, "fetch_source_page", return_value=b"{}") as fetch,
    ):
        actual = service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    assert actual == b"{}"
    fetch.assert_called_once_with(request, page_token=None)
    records = tuple(tmp_path.rglob("lockbox_usage_log.jsonl"))
    assert len(records) == 1
    assert "alpaca_ranker_daily_fixture" in records[0].read_text()


def test_daily_transport_serializes_fixed_request_without_reading_prices() -> None:
    seen = []

    class EmptyJSON(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"{}"

        async def aclose(self) -> None:
            return None

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "content-encoding": "identity"},
            stream=EmptyJSON(),
        )

    source = AlpacaProvider(api_key="test", api_secret="test")
    source._source_transport = httpx.MockTransport(respond)
    assert (
        source.fetch_source_page(AlpacaDailyFixtureRequest(symbol="KHC"), page_token=None) == b"{}"
    )
    assert len(seen) == 1
    assert dict(seen[0].url.params) == {
        "symbols": "KHC",
        "start": "2023-01-26T05:00:00Z",
        "end": "2023-04-24T04:00:00Z",
        "timeframe": "1Day",
        "adjustment": "all",
        "asof": "2023-04-24",
        "feed": "sip",
        "currency": "USD",
        "limit": "1000",
        "sort": "asc",
    }

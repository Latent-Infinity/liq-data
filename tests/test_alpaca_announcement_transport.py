"""Opaque transport controls, without invented financial observations."""

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from announcement_test_support import isolated_announcement_http as isolated_announcement_http
from pydantic import ValidationError

from liq.data.alpaca_announcement_source import (
    AlpacaAnnouncementCalendarRequest,
    AlpacaAnnouncementDailyRequest,
)
from liq.data.alpaca_source import AlpacaCalendarRequest, AlpacaCaptureRefused
from liq.data.exceptions import LockboxViolationError
from liq.data.lockbox import (
    CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1,
    CLASSICAL_RANKER_CALENDAR_REFERENCE_LEDGER_V1,
    CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1,
    LockboxLedger,
)
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.service import DataService


class OpaqueBody(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"x" * 131073


@pytest.mark.parametrize("day", [date(2020, 1, 1), date(2022, 4, 25)])
def test_subclass_cannot_override_frozen_slot(tmp_path: Path, day: date) -> None:
    class AlteredDaily(AlpacaAnnouncementDailyRequest):
        @property
        def start(self) -> datetime:
            return datetime(day.year, day.month, day.day, tzinfo=UTC)

    request = AlteredDaily(symbol="KHC", accession="0001637459-22-000056")
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(AlpacaCaptureRefused):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()
    assert not list(tmp_path.rglob("*usage*.jsonl"))
    source = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    seen: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        seen.append(wire)
        raise AssertionError("subclasses must refuse before transport")

    source._source_transport = httpx.MockTransport(respond)
    with pytest.raises(AlpacaCaptureRefused):
        source.fetch_source_page(request, page_token=None)
    assert seen == []


def test_missing_mock_is_refused_before_real_http() -> None:
    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    with pytest.raises(AssertionError, match="never use real HTTP"):
        provider.fetch_source_page(AlpacaAnnouncementCalendarRequest(), page_token=None)
    with httpx.Client() as client, pytest.raises(AssertionError, match="never use real HTTP"):
        client.get("https://example.invalid")


def test_calendar_subclass_refuses_before_provider(tmp_path: Path) -> None:
    class AlteredCalendar(AlpacaAnnouncementCalendarRequest):
        pass

    request = AlteredCalendar()
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
    )
    with (
        patch.object(service, "_get_provider") as provider,
        pytest.raises(AlpacaCaptureRefused, match="announcement_request_type"),
    ):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()
    source = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    with pytest.raises(AlpacaCaptureRefused, match="announcement_request_type"):
        source.fetch_source_page(request, page_token=None)


@pytest.mark.parametrize("calendar", [False, True])
@pytest.mark.parametrize("status", [200, 302, 401, 429, 503])
def test_transport_limits_apply_to_new_routes(calendar: bool, status: int) -> None:
    request = (
        AlpacaAnnouncementCalendarRequest()
        if calendar
        else AlpacaAnnouncementDailyRequest(symbol="SPY", accession="0001637459-23-000005")
    )
    seen: list[httpx.Request] = []
    attempts: list[bool] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        seen.append(wire)
        return httpx.Response(
            status,
            headers={"content-type": "application/json", "location": "https://example.invalid"},
            stream=OpaqueBody(),
        )

    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    provider._source_transport = httpx.MockTransport(respond)
    with pytest.raises(AlpacaCaptureRefused):
        provider.fetch_source_page(
            request, page_token=None, on_http_attempt=lambda: attempts.append(True)
        )
    assert len(seen) == len(attempts) == 1
    if not calendar:
        assert dict(seen[0].url.params) == {
            "symbols": "SPY",
            "start": "2023-02-14T05:00:00Z",
            "end": "2023-02-15T05:00:00Z",
            "timeframe": "1Day",
            "adjustment": "raw",
            "asof": "2023-02-15",
            "feed": "sip",
            "currency": "USD",
            "limit": "4",
            "sort": "asc",
        }
    with pytest.raises(AlpacaCaptureRefused, match="continuation"):
        provider.fetch_source_page(
            request, page_token="extra", on_http_attempt=lambda: attempts.append(True)
        )
    assert len(seen) == len(attempts) == 1


@pytest.mark.parametrize("calendar", [False, True])
@pytest.mark.parametrize(
    "ledger",
    [CLASSICAL_RANKER_CALENDAR_REFERENCE_LEDGER_V1, CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1],
)
def test_foreign_ledger_refuses_new_requests_before_credentials(
    tmp_path: Path, calendar: bool, ledger: LockboxLedger
) -> None:
    request = (
        AlpacaAnnouncementCalendarRequest()
        if calendar
        else AlpacaAnnouncementDailyRequest(symbol="KHC", accession="0001637459-22-000156")
    )
    service = DataService(data_root=tmp_path, lockbox_ledger=ledger)
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()
    assert not list(tmp_path.rglob("*usage*.jsonl"))


def test_announcement_ledger_does_not_admit_legacy_scope(tmp_path: Path) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaCalendarRequest(),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


@pytest.mark.parametrize(
    "changed",
    [
        {"max_pages": 2},
        {"max_response_bytes": 131073},
        {"limit": 5},
        {"currency": "EUR"},
        {"end": "2022-07-27"},
    ],
)
def test_constructed_request_cannot_expand_transport_budget(changed: dict[str, int | str]) -> None:
    request = AlpacaAnnouncementDailyRequest.model_construct(
        _fields_set={"symbol", "accession", *changed},
        **{"symbol": "KHC", "accession": "0001637459-22-000056", **changed},
    ).model_copy(update=changed)
    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    attempts: list[bool] = []
    seen: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        seen.append(wire)
        raise AssertionError("local refusal must precede transport")

    provider._source_transport = httpx.MockTransport(respond)
    with pytest.raises(ValidationError):
        provider.fetch_source_page(
            request, page_token=None, on_http_attempt=lambda: attempts.append(True)
        )
    assert attempts == []
    assert seen == []

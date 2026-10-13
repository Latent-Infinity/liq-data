"""Nonfinancial controls for finite announcement source request boundaries."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx
import pytest
from announcement_test_support import isolated_announcement_http as isolated_announcement_http
from pydantic import ValidationError

from liq.data.exceptions import LockboxViolationError
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.service import DataService


class MetadataControl(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"NONEXECUTING_METADATA_CONTROL":true}'


@pytest.mark.parametrize("symbol", ["KHC", "SPY"])
@pytest.mark.parametrize(
    "accession,first,last",
    [
        ("0001637459-22-000056", date(2022, 4, 26), date(2022, 4, 27)),
        ("0001637459-22-000156", date(2022, 7, 26), date(2022, 7, 27)),
        ("0001637459-22-000186", date(2022, 10, 25), date(2022, 10, 26)),
        ("0001637459-23-000005", date(2023, 2, 14), date(2023, 2, 15)),
    ],
)
def test_transport_uses_only_proposed_native_pair_when_accession_is_fixed(
    symbol: str, accession: str, first: date, last: date
) -> None:
    # Given: a metadata-only wire control, not an accepted financial response.
    from liq.data.alpaca_announcement_source import AlpacaAnnouncementDailyRequest

    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=MetadataControl(),
        )

    request = AlpacaAnnouncementDailyRequest.model_validate(
        {"symbol": symbol, "accession": accession}
    )
    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    provider._source_transport = httpx.MockTransport(respond)
    # When: the real HTTP adapter executes against the control transport.
    output = provider.fetch_source_page(request, page_token=None)
    # Then: only the frozen raw daily pair is requested; no financial value is consumed.
    assert output == b'{"NONEXECUTING_METADATA_CONTROL":true}'
    assert len(seen) == 1
    query = seen[0].url.params
    assert seen[0].url.path == "/v2/stocks/bars"
    assert query["symbols"] == symbol
    assert query["timeframe"] == "1Day"
    assert query["adjustment"] == "raw"
    assert query["feed"] == "sip"
    assert (
        datetime.fromisoformat(query["start"]).astimezone(ZoneInfo("America/New_York")).date()
        == first
    )
    assert (
        datetime.fromisoformat(query["end"]).astimezone(ZoneInfo("America/New_York")).date() == last
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"symbol": "META"},
        {"accession": "OTHER"},
        {"adjustment": "all"},
        {"feed": "iex"},
        {"asof": date(2023, 2, 16)},
        {"start": datetime(2022, 4, 25, tzinfo=UTC)},
    ],
)
def test_daily_request_refuses_new_cells_when_scope_is_changed(
    changed: dict[str, str | date | datetime],
) -> None:
    from liq.data.alpaca_announcement_source import AlpacaAnnouncementDailyRequest

    parameters = {"symbol": "KHC", "accession": "0001637459-22-000056", **changed}
    with pytest.raises(ValidationError):
        AlpacaAnnouncementDailyRequest.model_validate(parameters)


def test_guard_logs_exact_announcement_scope_when_context_is_authorized(tmp_path: Path) -> None:
    from liq.data.alpaca_announcement_source import AlpacaAnnouncementDailyRequest
    from liq.data.lockbox import CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1

    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
    )
    request = AlpacaAnnouncementDailyRequest(symbol="KHC", accession="0001637459-22-000056")
    # Given: guard logging is observable before the concrete provider is resolved.
    with (
        patch.object(
            service, "_get_provider", side_effect=RuntimeError("CONTROL_PROVIDER_BOUNDARY")
        ),
        pytest.raises(RuntimeError, match="CONTROL_PROVIDER_BOUNDARY"),
    ):
        # When: the exact characterization request crosses the guarded service.
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    # Then: the dedicated dataset and proposed two-day interval are recorded.
    logs = list(tmp_path.rglob("*usage*.jsonl"))
    assert len(logs) == 1
    entries = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["dataset"] == "alpaca_ranker_announcement_daily_fixture"
    assert entries[0]["start"] == "2022-04-26"
    assert entries[0]["end"] == "2022-04-27"


def test_default_ledger_refuses_announcement_read_when_unregistered(tmp_path: Path) -> None:
    from liq.data.alpaca_announcement_source import AlpacaAnnouncementDailyRequest

    service = DataService(data_root=tmp_path)
    request = AlpacaAnnouncementDailyRequest(symbol="KHC", accession="0001637459-22-000056")
    with (
        patch.object(
            service,
            "_get_provider",
            side_effect=AssertionError("transport must remain unreachable"),
        ),
        pytest.raises(LockboxViolationError),
    ):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )


def test_calendar_request_refuses_scope_changes_when_constructed() -> None:
    from liq.data.alpaca_announcement_source import AlpacaAnnouncementCalendarRequest

    with pytest.raises(ValidationError):
        AlpacaAnnouncementCalendarRequest(start=date(2022, 4, 24))


@pytest.mark.parametrize("calendar", [False, True])
@pytest.mark.parametrize(
    "purpose,arm,token,final",
    [
        ("dev_smoke", "classical_cross_sectional_ranker", None, False),
        ("characterization", "other", None, False),
        ("characterization", "classical_cross_sectional_ranker", "extra", False),
        ("characterization", "classical_cross_sectional_ranker", None, True),
    ],
)
def test_changed_authority_refuses_before_provider(
    tmp_path: Path, calendar: bool, purpose: str, arm: str, token: str | None, final: bool
) -> None:
    from liq.data.alpaca_announcement_source import (
        AlpacaAnnouncementCalendarRequest,
        AlpacaAnnouncementDailyRequest,
    )
    from liq.data.lockbox import CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1

    request = (
        AlpacaAnnouncementCalendarRequest()
        if calendar
        else AlpacaAnnouncementDailyRequest(symbol="KHC", accession="0001637459-22-000056")
    )
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            request, page_token=token, purpose=purpose, arm_id=arm, final_portfolio_review=final
        )
    provider.assert_not_called()
    assert not list(tmp_path.rglob("*usage*.jsonl"))


@pytest.mark.parametrize("calendar", [False, True])
def test_forged_instance_refuses_at_both_boundaries(tmp_path: Path, calendar: bool) -> None:
    from liq.data.alpaca_announcement_source import (
        AlpacaAnnouncementCalendarRequest,
        AlpacaAnnouncementDailyRequest,
    )
    from liq.data.lockbox import CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1

    request = (
        AlpacaAnnouncementCalendarRequest().model_copy(update={"end": date(2023, 2, 16)})
        if calendar
        else AlpacaAnnouncementDailyRequest(
            symbol="KHC", accession="0001637459-22-000056"
        ).model_copy(update={"accession": "OTHER"})
    )
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(ValidationError):
        service.fetch_alpaca_source_page(
            request,
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()
    seen: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        seen.append(wire)
        return httpx.Response(200, json={"NONEXECUTING_METADATA_CONTROL": True})

    source = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    source._source_transport = httpx.MockTransport(respond)
    with pytest.raises(ValidationError):
        source.fetch_source_page(request, page_token=None)
    assert seen == []


def test_calendar_route_and_guard_log_use_dedicated_scope(tmp_path: Path) -> None:
    from liq.data.alpaca_announcement_source import AlpacaAnnouncementCalendarRequest
    from liq.data.alpaca_source import AlpacaCaptureRefused
    from liq.data.lockbox import CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1, resolve_dataset

    seen: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        seen.append(wire)
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=MetadataControl()
        )

    source = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    source._source_transport = httpx.MockTransport(respond)
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1
    )
    with patch.object(service, "_get_provider", return_value=source):
        assert (
            service.fetch_alpaca_source_page(
                AlpacaAnnouncementCalendarRequest(),
                page_token=None,
                purpose="characterization",
                arm_id="classical_cross_sectional_ranker",
            )
            == b'{"NONEXECUTING_METADATA_CONTROL":true}'
        )
    assert len(seen) == 1
    assert (
        str(seen[0].url)
        == "https://paper-api.alpaca.markets/v2/calendar?start=2022-04-25&end=2023-02-15&date_type=TRADING"
    )
    entry = json.loads(next(tmp_path.rglob("*usage*.jsonl")).read_text())
    assert (entry["dataset"], entry["start"], entry["end"]) == (
        "alpaca_ranker_announcement_calendar_reference",
        "2022-04-25",
        "2023-02-15",
    )
    assert (
        resolve_dataset("databento", "KHC", data_kind="ranker_announcement_daily_fixture") is None
    )
    with pytest.raises(AlpacaCaptureRefused, match="continuation"):
        source.fetch_source_page(AlpacaAnnouncementCalendarRequest(), page_token="extra")
    assert len(seen) == 1

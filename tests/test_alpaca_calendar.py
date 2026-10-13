"""Nonfinancial controls for one bounded retail calendar reference request."""

import json
from datetime import UTC, date, datetime
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from liq.data.alpaca_calendar import CalendarRefused, parse_calendar
from liq.data.alpaca_source import AlpacaCalendarRequest, AlpacaCaptureRefused
from liq.data.exceptions import LockboxViolationError
from liq.data.lockbox import CLASSICAL_RANKER_CALENDAR_REFERENCE_LEDGER_V1, resolve_dataset
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.service import DataService


def test_calendar_scope_is_fixed() -> None:
    request = AlpacaCalendarRequest()
    assert (request.start, request.end) == (date(2023, 1, 26), date(2023, 4, 24))
    for changed in ({"start": date(2023, 1, 25)}, {"end": date(2023, 4, 25)}, {"max_pages": 2}):
        with pytest.raises(ValidationError):
            AlpacaCalendarRequest(**changed)


def test_calendar_routes_only_to_its_alpaca_ledger() -> None:
    assert (
        resolve_dataset("alpaca", "US_MARKET", data_kind="ranker_calendar_reference")
        == "alpaca_ranker_calendar_reference"
    )
    assert resolve_dataset("databento", "US_MARKET", data_kind="ranker_calendar_reference") is None
    assert resolve_dataset("alpaca", "US_MARKET") is None


def test_calendar_guard_logs_dedicated_reference_before_provider(tmp_path) -> None:
    service = DataService(
        data_root=tmp_path,
        research_log_root=tmp_path,
        lockbox_ledger=CLASSICAL_RANKER_CALENDAR_REFERENCE_LEDGER_V1,
    )
    provider = AlpacaProvider(api_key="nonsecret-key", api_secret="nonsecret-secret")

    def page(*args, **kwargs):
        log = json.loads((tmp_path / "lockbox_usage_log.jsonl").read_text())
        assert log["dataset"] == "alpaca_ranker_calendar_reference"
        assert log["purpose"] == "characterization"
        return b"[]"

    with (
        patch.object(service, "_get_provider", return_value=provider),
        patch.object(provider, "fetch_source_page", side_effect=page),
    ):
        assert (
            service.fetch_alpaca_source_page(
                AlpacaCalendarRequest(),
                page_token=None,
                purpose="characterization",
                arm_id="classical_cross_sectional_ranker",
            )
            == b"[]"
        )


@pytest.mark.parametrize(
    "purpose,arm_id,page_token",
    [
        ("discovery", "classical_cross_sectional_ranker", None),
        ("characterization", "other", None),
        ("characterization", "classical_cross_sectional_ranker", "next"),
    ],
)
def test_calendar_guard_refuses_before_provider(tmp_path, purpose, arm_id, page_token) -> None:
    service = DataService(
        data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_CALENDAR_REFERENCE_LEDGER_V1
    )
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaCalendarRequest(), page_token=page_token, purpose=purpose, arm_id=arm_id
        )
    provider.assert_not_called()


def test_calendar_wrong_ledger_refuses_before_provider(tmp_path) -> None:
    service = DataService(data_root=tmp_path)
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaCalendarRequest(),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
        )
    provider.assert_not_called()


def test_calendar_zero_transport_attempt_on_local_refusal() -> None:
    calls: list[str] = []
    provider = AlpacaProvider(api_key="nonsecret-key", api_secret="nonsecret-secret")
    with pytest.raises(AlpacaCaptureRefused, match="calendar_continuation_forbidden"):
        provider.fetch_source_page(
            AlpacaCalendarRequest(),
            page_token="next",
            on_http_attempt=lambda: calls.append("attempt"),
        )
    assert calls == []


def test_calendar_attempt_witness_is_at_transport_entry() -> None:
    calls: list[str] = []

    def transport(request):
        assert calls == ["attempt"]
        return httpx.Response(403)

    provider = AlpacaProvider(api_key="nonsecret-key", api_secret="nonsecret-secret")
    provider._source_transport = httpx.MockTransport(transport)
    with pytest.raises(AlpacaCaptureRefused):
        provider.fetch_source_page(
            AlpacaCalendarRequest(),
            page_token=None,
            on_http_attempt=lambda: calls.append("attempt"),
        )
    assert calls == ["attempt"]


def test_calendar_guard_refusal_never_calls_attempt_witness(tmp_path) -> None:
    calls: list[str] = []
    service = DataService(data_root=tmp_path)
    with pytest.raises(LockboxViolationError):
        service.fetch_alpaca_source_page(
            AlpacaCalendarRequest(),
            page_token=None,
            purpose="characterization",
            arm_id="classical_cross_sectional_ranker",
            on_http_attempt=lambda: calls.append("attempt"),
        )
    assert calls == []


def test_retail_capture_single_fixed_get() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, stream=httpx.ByteStream(b"[]"), headers={"content-type": "application/json"}
        )

    provider = AlpacaProvider(api_key="nonsecret-key", api_secret="nonsecret-secret")
    provider._source_transport = httpx.MockTransport(handle)
    assert provider.fetch_source_page(AlpacaCalendarRequest(), page_token=None) == b"[]"
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "GET"
    assert request.url.host == "paper-api.alpaca.markets"
    assert request.url.path == "/v2/calendar"
    assert dict(request.url.params) == {
        "start": "2023-01-26",
        "end": "2023-04-24",
        "date_type": "TRADING",
    }
    assert request.headers["accept-encoding"] == "identity"
    assert set(request.extensions["timeout"].values()) == {20.0}


@pytest.mark.parametrize(
    "headers,body",
    [
        ({"content-type": "application/json", "content-encoding": "gzip"}, b"[]"),
        ({"content-type": "text/html"}, b"[]"),
        ({"content-type": "application/json"}, b"x" * 1048577),
    ],
)
def test_calendar_transport_bounds(headers: dict[str, str], body: bytes) -> None:
    provider = AlpacaProvider(api_key="nonsecret-key", api_secret="nonsecret-secret")
    provider._source_transport = httpx.MockTransport(
        lambda _: httpx.Response(200, headers=headers, stream=httpx.ByteStream(body))
    )
    with pytest.raises(AlpacaCaptureRefused):
        provider.fetch_source_page(AlpacaCalendarRequest(), page_token=None)


@pytest.mark.parametrize(
    "dates",
    [
        (),
        (date(2023, 1, 27), date(2023, 1, 26)),
        (date(2023, 1, 26), date(2023, 1, 26)),
        (date(2023, 1, 25),),
    ],
)
def test_calendar_invalid_expected_labels(dates: tuple[date, ...]) -> None:
    with pytest.raises(CalendarRefused, match="invalid_expected_dates"):
        parse_calendar(b"[]", datetime(2026, 10, 3, tzinfo=UTC), dates)


def test_calendar_capture_clock_and_decoder_budget() -> None:
    with pytest.raises(CalendarRefused, match="invalid_capture_clock"):
        parse_calendar(b"[]", datetime(2026, 10, 3), (date(2023, 1, 26),))
    with pytest.raises(CalendarRefused, match="byte_budget"):
        parse_calendar(b"x" * 1048577, datetime(2026, 10, 3, tzinfo=UTC), (date(2023, 1, 26),))


@pytest.mark.parametrize("status", [302, 403, 429, 500])
def test_calendar_http_refusal_never_retries(status: int) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            status, headers={"location": "https://api.alpaca.markets/v2/calendar"}
        )

    provider = AlpacaProvider(api_key="nonsecret-key", api_secret="nonsecret-secret")
    provider._source_transport = httpx.MockTransport(handle)
    with pytest.raises(AlpacaCaptureRefused) as exc:
        provider.fetch_source_page(AlpacaCalendarRequest(), page_token=None)
    assert exc.value.status_code == status
    assert len(seen) == 1


def test_calendar_continuation_refused_before_transport() -> None:
    provider = AlpacaProvider(api_key="nonsecret-key", api_secret="nonsecret-secret")
    with pytest.raises(AlpacaCaptureRefused, match="calendar_continuation_forbidden"):
        provider.fetch_source_page(AlpacaCalendarRequest(), page_token="next")


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{}",
        b"null",
        b"[{}]",
        b'[{"date":"2023-01-26","date":"2023-01-27","open":"09:30","close":"16:00"}]',
    ],
)
def test_calendar_invalid_schema_refused(raw: bytes) -> None:
    with pytest.raises(CalendarRefused):
        parse_calendar(raw, datetime(2026, 10, 3, tzinfo=UTC), (date(2023, 1, 26),))


@pytest.mark.parametrize(
    "change",
    [
        {"date": "2023-01-25"},
        {"open": "09:30Z"},
        {"close": "xx"},
        {"open": "16:00"},
        {"date": "2023-1-26"},
    ],
)
def test_calendar_bad_labels_refused(change: dict[str, str]) -> None:
    row = {"date": "2023-01-26", "open": "09:30", "close": "16:00", **change}
    with pytest.raises(CalendarRefused):
        parse_calendar(
            json.dumps([row]).encode(), datetime(2026, 10, 3, tzinfo=UTC), (date(2023, 1, 26),)
        )


def test_calendar_missing_required_session_refused() -> None:
    raw = b'[{"date":"2023-01-26","open":"09:30","close":"16:00"}]'
    with pytest.raises(CalendarRefused, match="date_labels_disagree"):
        parse_calendar(
            raw, datetime(2026, 10, 3, tzinfo=UTC), (date(2023, 1, 26), date(2023, 1, 27))
        )


def test_calendar_preserves_labels_and_utc_conversion() -> None:
    # Calendar clock metadata, not fabricated price/signal/regime inputs.
    raw = b'[{"date":"2023-01-26","open":"09:30","close":"16:00"},{"date":"2023-04-24","open":"09:30","close":"16:00"}]'
    result = parse_calendar(
        raw, datetime(2026, 10, 3, tzinfo=UTC), (date(2023, 1, 26), date(2023, 4, 24))
    )
    assert result.timezone == "America/New_York"
    assert result.sessions[0].original_open == "09:30"
    assert result.sessions[0].open_utc == datetime(2023, 1, 26, 14, 30, tzinfo=UTC)
    assert result.sessions[1].open_utc == datetime(2023, 4, 24, 13, 30, tzinfo=UTC)

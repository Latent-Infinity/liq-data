"""Nonfinancial calendar and negative close-schema controls; no price evidence."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from liq.data.alpaca_calendar import CalendarRefused
from liq.data.alpaca_daily_fixture import DailyFixtureRefused
from liq.data.alpaca_momentum_fixture import parse_momentum_calendar, parse_momentum_daily
from liq.data.alpaca_momentum_source import (
    AlpacaMomentumDailyRequest,
)

CAPTURED = datetime(2026, 10, 3, tzinfo=UTC)


DAYS = (
    "2022-04-29",
    "2022-05-31",
    "2022-06-30",
    "2022-07-29",
    "2022-08-31",
    "2022-09-30",
    "2022-10-31",
    "2022-11-30",
    "2022-12-30",
    "2023-01-31",
    "2023-02-28",
    "2023-03-31",
)


def calendar_body(days: tuple[str, ...] = DAYS) -> bytes:
    return json.dumps([{"date": day, "open": "09:30", "close": "16:00"} for day in days]).encode()


def test_calendar_mapping_keeps_native_dst_labels_separate_from_close() -> None:
    decoded = parse_momentum_calendar(calendar_body(), CAPTURED)
    assert decoded.anchor_month == date(2023, 4, 1)
    assert decoded.monthly_endpoints == decoded.expected_labels
    assert decoded.expected_labels[0] == datetime(2022, 4, 29, 4, tzinfo=UTC)
    assert decoded.expected_labels[7] == datetime(2022, 11, 30, 5, tzinfo=UTC)
    assert decoded.expected_labels[-1] == datetime(2023, 3, 31, 4, tzinfo=UTC)
    assert decoded.reference.sessions[-1].close_utc == datetime(2023, 3, 31, 20, tzinfo=UTC)


def test_mapping_selects_last_declared_date_without_filling_calendar() -> None:
    decoded = parse_momentum_calendar(calendar_body(("2022-04-28", *DAYS)), CAPTURED)
    assert len(decoded.expected_labels) == 13
    assert len(decoded.monthly_endpoints) == 12
    assert decoded.monthly_endpoints[0] == datetime(2022, 4, 29, 4, tzinfo=UTC)


@pytest.mark.parametrize(
    "raw",
    [
        b"not JSON",
        b"\xff",
        b"{}",
        b"[]",
        b'[{"date":"2022-04-29","date":"2022-04-29","open":"09:30","close":"16:00"}]',
        calendar_body(DAYS[:-1]),
        calendar_body((DAYS[0], *DAYS)),
        calendar_body(tuple(reversed(DAYS))),
        calendar_body(("2022-03-31", *DAYS)),
        calendar_body(("2022-04-30", *DAYS[1:])),
        calendar_body().replace(b"09:30", b"24:00", 1),
        calendar_body().replace(b"16:00", b"09:00", 1),
    ],
)
def test_calendar_anomalies_refuse_without_repair(raw: bytes) -> None:
    with pytest.raises(CalendarRefused):
        parse_momentum_calendar(raw, CAPTURED)


@pytest.mark.parametrize("clock", [datetime(2023, 3, 30, tzinfo=UTC), datetime(2026, 10, 3)])
def test_calendar_requires_completed_utc_capture(clock: datetime) -> None:
    with pytest.raises(CalendarRefused):
        parse_momentum_calendar(calendar_body(), clock)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"not JSON",
        b"\xff",
        b'{"bars":{},"bars":{}}',
        b'{"bars":{"KHC":[]},"next_page_token":"continue"}',
        b'{"bars":{"SPY":[]}}',
        b'{"bars":{"KHC":[]}}',
        b'{"bars":{"KHC":[{"t":"2022-04-29T04:00:00Z","c":0}]}}',
    ],
)
def test_daily_negative_schema_controls_use_no_positive_prices(raw: bytes) -> None:
    calendar = parse_momentum_calendar(calendar_body(), CAPTURED)
    with pytest.raises(DailyFixtureRefused):
        parse_momentum_daily(raw, AlpacaMomentumDailyRequest(), calendar, CAPTURED)


def test_decoder_byte_and_clock_limits_precede_schema() -> None:
    with pytest.raises(CalendarRefused, match="byte"):
        parse_momentum_calendar(b" " * 131073, CAPTURED)
    calendar = parse_momentum_calendar(calendar_body(), CAPTURED)
    with pytest.raises(DailyFixtureRefused, match="byte"):
        parse_momentum_daily(b" " * 131073, AlpacaMomentumDailyRequest(), calendar, CAPTURED)
    with pytest.raises(DailyFixtureRefused, match="clock"):
        parse_momentum_daily(
            b"{}", AlpacaMomentumDailyRequest(), calendar, CAPTURED - timedelta(days=1)
        )


def test_calendar_record_limit_and_noncanonical_date() -> None:
    with pytest.raises(CalendarRefused, match="record_budget"):
        parse_momentum_calendar(calendar_body((DAYS[0],) * 401), CAPTURED)
    for day in ("20220429", "not-a-date"):
        with pytest.raises(CalendarRefused):
            parse_momentum_calendar(calendar_body((day, *DAYS[1:])), CAPTURED)


def test_daily_refuses_changed_calendar_mapping_before_prices() -> None:
    calendar = parse_momentum_calendar(calendar_body(), CAPTURED)
    changed = replace(calendar, monthly_endpoints=tuple(reversed(calendar.monthly_endpoints)))
    with pytest.raises(DailyFixtureRefused, match="calendar_mapping"):
        parse_momentum_daily(b"{}", AlpacaMomentumDailyRequest(), changed, CAPTURED)

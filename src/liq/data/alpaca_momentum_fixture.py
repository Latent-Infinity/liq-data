"""Strict provider-declared annual calendar and native daily-close reconciliation."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from itertools import pairwise
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter, ValidationError

from liq.data.alpaca_calendar import (
    CalendarReference,
    CalendarRefused,
    CalendarSession,
    _clock,
    _Row,
    _unique_keys,
)
from liq.data.alpaca_daily_fixture import DailyFixtureRefused, _Page, _unique_object
from liq.data.alpaca_momentum_source import (
    MOMENTUM_END,
    MOMENTUM_MONTHS,
    MOMENTUM_START,
    AlpacaMomentumCalendarRequest,
    AlpacaMomentumDailyRequest,
)


@dataclass(frozen=True, slots=True)
class MomentumCalendar:
    reference: CalendarReference
    expected_labels: tuple[datetime, ...]
    monthly_endpoints: tuple[datetime, ...]
    anchor_month: date = date(2023, 4, 1)


@dataclass(frozen=True, slots=True)
class MomentumDailyFixture:
    request: AlpacaMomentumDailyRequest
    calendar: MomentumCalendar
    original_labels: tuple[str, ...]
    closes: tuple[Decimal, ...]
    source_sha256: str
    available_at: datetime


def _mapping(
    sessions: tuple[CalendarSession, ...],
) -> tuple[tuple[datetime, ...], tuple[datetime, ...]]:
    dates = tuple(session.date for session in sessions)
    if (
        not dates
        or len(dates) > 400
        or any(not MOMENTUM_START <= day <= MOMENTUM_END for day in dates)
        or any(day.weekday() >= 5 for day in dates)
        or any(left >= right for left, right in pairwise(dates))
        or {(day.year, day.month) for day in dates} != set(MOMENTUM_MONTHS)
    ):
        raise CalendarRefused("momentum_calendar_dates")
    labels = tuple(
        datetime.combine(day, time(), ZoneInfo("America/New_York")).astimezone(UTC) for day in dates
    )
    last_by_month = {(day.year, day.month): label for day, label in zip(dates, labels, strict=True)}
    return labels, tuple(last_by_month[month] for month in MOMENTUM_MONTHS)


def parse_momentum_calendar(raw: bytes, captured_at: datetime) -> MomentumCalendar:
    """Retained bytes declare coverage; no reconstruction of omitted sessions."""
    request = AlpacaMomentumCalendarRequest()
    if len(raw) > request.max_response_bytes:
        raise CalendarRefused("momentum_calendar_byte_budget")
    if captured_at.utcoffset() != timedelta(0) or captured_at.date() < request.end:
        raise CalendarRefused("momentum_calendar_capture_clock")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_keys)
        rows = TypeAdapter(list[_Row]).validate_python(payload, strict=True)
    except CalendarRefused:
        raise
    except (ValueError, UnicodeDecodeError, ValidationError) as exc:
        raise CalendarRefused("momentum_calendar_schema") from exc
    if not rows or len(rows) > 400:
        raise CalendarRefused("momentum_calendar_record_budget")
    sessions = []
    for row in rows:
        try:
            day = date.fromisoformat(row.date)
        except ValueError as exc:
            raise CalendarRefused("momentum_calendar_date") from exc
        if day.isoformat() != row.date:
            raise CalendarRefused("momentum_calendar_date_label")
        opened, closed = _clock(day, row.open), _clock(day, row.close)
        if opened >= closed or closed > captured_at:
            raise CalendarRefused("momentum_calendar_session_interval")
        sessions.append(CalendarSession(day, row.open, row.close, opened, closed))
    reference = CalendarReference(tuple(sessions), hashlib.sha256(raw).hexdigest(), captured_at)
    labels, endpoints = _mapping(reference.sessions)
    return MomentumCalendar(reference, labels, endpoints)


def parse_momentum_daily(
    raw: bytes,
    request: AlpacaMomentumDailyRequest,
    calendar: MomentumCalendar,
    captured_at: datetime,
) -> MomentumDailyFixture:
    """Match every declared trading date, preserving c and its native day-start t."""
    request = AlpacaMomentumDailyRequest.model_validate(request)
    if len(raw) > request.max_response_bytes:
        raise DailyFixtureRefused("momentum_daily_byte_budget")
    if (
        captured_at.utcoffset() != timedelta(0)
        or captured_at < calendar.reference.captured_at
        or calendar.reference.captured_at.utcoffset() != timedelta(0)
    ):
        raise DailyFixtureRefused("momentum_daily_capture_clock")
    labels, endpoints = _mapping(calendar.reference.sessions)
    if (labels, endpoints, date(2023, 4, 1)) != (
        calendar.expected_labels,
        calendar.monthly_endpoints,
        calendar.anchor_month,
    ):
        raise DailyFixtureRefused("momentum_daily_calendar_mapping")
    try:
        payload = json.loads(
            raw, parse_float=Decimal, parse_int=Decimal, object_pairs_hook=_unique_object
        )
        page = _Page.model_validate(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise DailyFixtureRefused("momentum_daily_schema") from exc
    if page.next_page_token is not None:
        raise DailyFixtureRefused("momentum_daily_continuation")
    if set(page.bars) != {request.symbol}:
        raise DailyFixtureRefused("momentum_daily_symbol")
    rows = page.bars[request.symbol]
    if len(rows) != len(labels):
        raise DailyFixtureRefused("momentum_daily_complete_calendar_required")
    original = tuple(row.t for row in rows)
    if original != tuple(label.isoformat().replace("+00:00", "Z") for label in labels):
        raise DailyFixtureRefused("momentum_daily_native_labels")
    return MomentumDailyFixture(
        request,
        calendar,
        original,
        tuple(row.c for row in rows),
        hashlib.sha256(raw).hexdigest(),
        captured_at,
    )

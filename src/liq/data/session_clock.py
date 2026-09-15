"""XNYS bar-start clocks, with explicit missing target minutes on early closes."""

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Final
from zoneinfo import ZoneInfo

import exchange_calendars as ec

_EASTERN: Final = ZoneInfo("America/New_York")
_MINUTE: Final = timedelta(minutes=1)


class SessionClockError(ValueError):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@lru_cache(maxsize=32)
def _calendar(year: int) -> ec.ExchangeCalendar:
    return ec.get_calendar("XNYS", start=f"{year}-01-01", end=f"{year}-12-31")


@lru_cache(maxsize=16384)
def _bounds(day: date) -> tuple[datetime, datetime] | None:
    calendar = _calendar(day.year)
    label = day.isoformat()
    if day < calendar.first_session.date() or day > calendar.last_session.date():
        return None
    if not calendar.is_session(label):
        return None
    return (
        calendar.session_open(label).to_pydatetime().astimezone(UTC),
        calendar.session_close(label).to_pydatetime().astimezone(UTC),
    )


def _require_utc_minute(timestamp: datetime) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise SessionClockError("timestamp must be UTC-aware")
    if timestamp.second or timestamp.microsecond:
        raise SessionClockError("timestamp must have minute precision")


@lru_cache(maxsize=16384)
def _shift_day(day: date, sessions: int) -> date:
    step = timedelta(days=1 if sessions > 0 else -1)
    cursor = day
    remaining = abs(sessions)
    while remaining:
        cursor += step
        if _bounds(cursor) is not None:
            remaining -= 1
    return cursor


def shift_session_minute(timestamp: datetime, sessions: int) -> datetime | None:
    """Shift a UTC regular bar start by sessions, preserving its New York clock.

    Negative offsets move backwards. Invalid source minutes raise SessionClockError;
    a target session that closes before the requested minute returns None.
    """
    _require_utc_minute(timestamp)
    if isinstance(sessions, bool) or not isinstance(sessions, int):
        raise SessionClockError("sessions must be an integer")
    local = timestamp.astimezone(_EASTERN)
    bounds = _bounds(local.date())
    if bounds is None or not bounds[0] <= timestamp < bounds[1]:
        raise SessionClockError("timestamp is not a regular-session bar start")
    target_day = _shift_day(local.date(), sessions)
    target = datetime.combine(target_day, local.time(), _EASTERN).astimezone(UTC)
    target_bounds = _bounds(target_day)
    if target_bounds is None or not target_bounds[0] <= target < target_bounds[1]:
        return None
    return target


def regular_session_minutes(start: datetime, end: datetime) -> tuple[datetime, ...]:
    """Return expected UTC bar starts in [start, end), excluding closed minutes."""
    _require_utc_minute(start)
    _require_utc_minute(end)
    if end < start:
        raise SessionClockError("end must be at or after start")
    minutes: list[datetime] = []
    day = start.date()
    while day <= end.date():
        bounds = _bounds(day)
        if bounds is not None:
            cursor, stop = max(start, bounds[0]), min(end, bounds[1])
            while cursor < stop:
                minutes.append(cursor)
                cursor += _MINUTE
        day += timedelta(days=1)
    return tuple(minutes)

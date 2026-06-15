"""Trading-calendar helpers (NYSE/XNYS).

Centralizes ``exchange_calendars`` usage so consumers (the scanner,
the sweep loop) do not each have to know about the package directly.
Functions accept UTC ``datetime`` inputs and return UTC ``datetime``
outputs — no naive timestamps.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import exchange_calendars as ec

GapClass = Literal[
    "weeknight",
    "weekend",
    "pre_holiday",
    "post_holiday",
    "long_holiday",
    "halt_reopen",
]

_DEFAULT_VENUE = "XNYS"
_MINUTES_PER_REGULAR_SESSION = 390
_EASTERN = ZoneInfo("America/New_York")
_EXTENDED_OPEN = time(4, 0)
_EXTENDED_CLOSE = time(20, 0)


def _calendar(venue: str = _DEFAULT_VENUE) -> ec.ExchangeCalendar:
    """Return the cached ``exchange_calendars`` instance for ``venue``."""
    return ec.get_calendar(venue)


def trading_minutes_window(end: datetime, n: int) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` covering the previous ``n`` trading minutes.

    The window is half-open ``[start, end)`` on the *trading-minute*
    grid: weekends, holidays, and out-of-session minutes are stepped
    over. ``end`` is returned verbatim so the caller's intent
    survives — typically ``end`` is a session-close timestamp.
    """
    if n <= 0:
        raise ValueError("trading_minutes_window: n must be positive")

    cal = _calendar()
    # Walk back far enough to definitely contain n trading minutes.
    # ~390 trading minutes/session, so days needed = ceil(n / 390) * 2
    # plus a 7-day pad for weekends/holidays.
    sessions_needed = max(1, (n // _MINUTES_PER_REGULAR_SESSION) + 1)
    lookback = sessions_needed * 2 + 7
    probe_start = end - timedelta(days=lookback)

    # ``minutes_in_range`` is inclusive on both ends.
    index = cal.minutes_in_range(probe_start, end)
    # Half-open: exclude any timestamps equal to ``end``.
    filtered = [ts.to_pydatetime().astimezone(UTC) for ts in index if ts.to_pydatetime() < end]
    if len(filtered) < n:
        raise ValueError(
            f"trading_minutes_window: only {len(filtered)} minutes available "
            f"in the {lookback}-day lookback ending {end!r}"
        )
    return filtered[-n], end


def extended_trading_minutes_window(end: datetime, n: int) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` covering previous extended-hours minutes.

    The equities extended-hours grid is 04:00-20:00 New York time on
    valid XNYS sessions. Holidays and weekends are skipped, and DST is
    handled by constructing the local session grid before converting to
    UTC.
    """
    if n <= 0:
        raise ValueError("extended_trading_minutes_window: n must be positive")

    cal = _calendar()
    sessions_needed = max(1, (n // (16 * 60)) + 1)
    lookback = sessions_needed * 2 + 7
    probe_start = end - timedelta(days=lookback)

    minutes: list[datetime] = []
    sessions = cal.sessions_in_range(probe_start.date().isoformat(), end.date().isoformat())
    for session in sessions:
        session_date = session.to_pydatetime().date()
        cursor = datetime.combine(session_date, _EXTENDED_OPEN, tzinfo=_EASTERN).astimezone(UTC)
        stop = datetime.combine(session_date, _EXTENDED_CLOSE, tzinfo=_EASTERN).astimezone(UTC)
        while cursor < stop:
            if probe_start <= cursor < end:
                minutes.append(cursor)
            cursor += timedelta(minutes=1)

    if len(minutes) < n:
        raise ValueError(
            f"extended_trading_minutes_window: only {len(minutes)} minutes available "
            f"in the {lookback}-day lookback ending {end!r}"
        )
    return minutes[-n], end


def trading_sessions_window(end: datetime, n: int) -> tuple[datetime, datetime]:
    """Return ``(start, end)`` covering the previous ``n`` trading sessions.

    ``start`` is the open of the earliest session in the range; ``end``
    is returned verbatim.
    """
    if n <= 0:
        raise ValueError("trading_sessions_window: n must be positive")
    cal = _calendar()
    end_session = cal.minute_to_session(end, direction="previous")
    if n == 1:
        first_session = end_session
    else:
        sessions = cal.sessions_window(end_session, -n)
        first_session = sessions[0]
    open_minute = cal.session_open(first_session)
    return open_minute.to_pydatetime().astimezone(UTC), end


def nyse_session_close(session: date) -> datetime:
    """Return the regular-session close timestamp (UTC) for ``session``."""
    cal = _calendar()
    close = cal.session_close(session.isoformat())
    return close.to_pydatetime().astimezone(UTC)


def closed_hours_between(c_prev: datetime, o: datetime) -> float:
    """Return the elapsed hours between a previous session close and the next open.

    Both inputs must be timezone-aware. ``c_prev`` is the prior close
    timestamp and ``o`` is the next open timestamp; the function returns
    the raw closed duration ``(o - c_prev)`` in hours. Calendar-derived
    in the sense that callers typically pass actual session boundaries
    obtained from this module (e.g. ``nyse_session_close``), but the
    arithmetic itself is a UTC subtraction — DST and calendar nuances
    are absorbed by the boundaries the caller supplies.

    Used by the volatility decomposition to populate
    ``closed_hours_t``, which feeds the per-closed-hour gap diagnostic
    documented in the canonical risk-variance research plan §3.1a.
    """
    if c_prev.tzinfo is None or o.tzinfo is None:
        raise ValueError("closed_hours_between: both timestamps must be tz-aware")
    if o < c_prev:
        raise ValueError("closed_hours_between: open must be at or after previous close")
    return (o - c_prev).total_seconds() / 3600.0


def classify_gap(c_prev: datetime, o: datetime) -> GapClass:
    """Classify the closed-market gap between a previous close and next open.

    Returns one of the labels in ``GapClass``. Both inputs must be
    timezone-aware. The label is derived from the XNYS trading calendar:

    - ``halt_reopen``: ``c_prev`` and ``o`` belong to the same trading
      session (intraday halt and resume).
    - ``weeknight``: adjacent trading sessions with no intervening
      calendar days.
    - ``weekend``: pure Friday→Monday gap with only weekend days
      interior and no market holidays.
    - ``pre_holiday``: a mid-week holiday gap where the holiday is
      immediately adjacent to ``c_prev`` (e.g. Wed→Fri with Thu
      holiday). The session at ``c_prev`` was the last trading day
      before the holiday.
    - ``post_holiday``: a mid-week holiday gap where the holiday is
      adjacent to ``o`` but not to ``c_prev`` (rare in practice; reserved
      for completeness).
    - ``long_holiday``: any gap that mixes weekend and holiday days,
      or that contains multiple holidays / interior trading sessions.

    Feeds ``gap_class_t`` on the volatility decomposition; never used
    to scale the canonical scalar (research plan §3.1a).
    """
    if c_prev.tzinfo is None or o.tzinfo is None:
        raise ValueError("classify_gap: both timestamps must be tz-aware")
    if o < c_prev:
        raise ValueError("classify_gap: open must be at or after previous close")

    cal = _calendar()
    c_prev_session_ts = cal.minute_to_session(c_prev, direction="previous")
    o_session_ts = cal.minute_to_session(o, direction="next")
    c_prev_session = c_prev_session_ts.to_pydatetime().date()
    o_session = o_session_ts.to_pydatetime().date()

    if c_prev_session == o_session:
        return "halt_reopen"

    days_between = (o_session - c_prev_session).days
    if days_between == 1:
        return "weeknight"

    interior_dates = [c_prev_session + timedelta(days=i) for i in range(1, days_between)]
    interior_sessions: list[date] = []
    weekend_days: list[date] = []
    holiday_days: list[date] = []
    for d in interior_dates:
        if cal.is_session(d.isoformat()):
            interior_sessions.append(d)
        elif d.weekday() >= 5:
            weekend_days.append(d)
        else:
            holiday_days.append(d)

    if interior_sessions:
        # A trading session in the interior means the caller passed a
        # multi-session gap — fold it into ``long_holiday`` as the most
        # conservative label.
        return "long_holiday"

    if weekend_days and holiday_days:
        return "long_holiday"
    if weekend_days:
        return "weekend"
    # Pure holiday days interior; pick pre vs post by which side the
    # first holiday is adjacent to.
    day_after_c_prev = c_prev_session + timedelta(days=1)
    if holiday_days[0] == day_after_c_prev:
        return "pre_holiday"
    return "post_holiday"


__all__ = [
    "GapClass",
    "classify_gap",
    "closed_hours_between",
    "extended_trading_minutes_window",
    "nyse_session_close",
    "trading_minutes_window",
    "trading_sessions_window",
]

"""Trading-calendar helpers (NYSE/XNYS).

Centralizes ``exchange_calendars`` usage so consumers (the scanner,
the sweep loop) do not each have to know about the package directly.
Functions accept UTC ``datetime`` inputs and return UTC ``datetime``
outputs — no naive timestamps.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as ec

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


__all__ = [
    "extended_trading_minutes_window",
    "nyse_session_close",
    "trading_minutes_window",
    "trading_sessions_window",
]

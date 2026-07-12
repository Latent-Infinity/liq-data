"""FX session tagging: Asian range, London open, and the FX trading week.

All boundaries are defined in local time and converted to UTC via
``zoneinfo`` so they track DST automatically (the same pattern as
``calendar.py``). Consumers work in UTC and convert only at display
boundaries.

- **Asian range** = ``[00:00, 08:00)`` Europe/London on the session date.
- **London open** = 08:00 Europe/London (07:00 UTC in BST, 08:00 UTC in GMT).
- **FX trading week** = ``[Sunday 17:00, Friday 17:00)`` America/New_York
  (the industry-standard 5pm-New-York roll; ~21:00-22:00 UTC depending on
  US DST). Saturday, Sunday before the open, and Friday after the close are
  closed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

_LONDON = "Europe/London"
_NEW_YORK = "America/New_York"

_LONDON_TZ = ZoneInfo(_LONDON)
_NEW_YORK_TZ = ZoneInfo(_NEW_YORK)

LONDON_OPEN = time(8, 0)
ASIAN_RANGE_START = time(0, 0)
LONDON_OPEN_FIRST_MINUTES = 15
FX_WEEK_ROLL = time(17, 0)  # 5pm New York


def london_open_utc(session_date: date) -> datetime:
    """London open (08:00 Europe/London) on ``session_date`` in UTC."""
    local = datetime.combine(session_date, LONDON_OPEN, tzinfo=_LONDON_TZ)
    return local.astimezone(UTC)


def asian_range_window_utc(session_date: date) -> tuple[datetime, datetime]:
    """Half-open ``[00:00, 08:00)`` Europe/London window on ``session_date``."""
    start = datetime.combine(session_date, ASIAN_RANGE_START, tzinfo=_LONDON_TZ)
    return start.astimezone(UTC), london_open_utc(session_date)


def london_open_window_utc(
    session_date: date, minutes: int = LONDON_OPEN_FIRST_MINUTES
) -> tuple[datetime, datetime]:
    """Half-open window covering the first ``minutes`` after the London open."""
    if minutes <= 0:
        raise ValueError(f"minutes must be positive, got {minutes}")
    start = london_open_utc(session_date)
    return start, start + timedelta(minutes=minutes)


def fx_session_date(ts: datetime) -> date:
    """London-local calendar date of a UTC (or tz-aware) timestamp."""
    if ts.tzinfo is None:
        raise ValueError("fx_session_date requires a tz-aware datetime")
    return ts.astimezone(_LONDON_TZ).date()


def is_fx_trading_week(ts: datetime) -> bool:
    """True if ``ts`` falls inside the ``[Sun 17:00, Fri 17:00)`` NY week."""
    if ts.tzinfo is None:
        raise ValueError("is_fx_trading_week requires a tz-aware datetime")
    local = ts.astimezone(_NEW_YORK_TZ)
    weekday = local.weekday()  # Mon=0 .. Sun=6
    if weekday == 5:  # Saturday
        return False
    if weekday == 6:  # Sunday: open only from 17:00
        return local.time() >= FX_WEEK_ROLL
    if weekday == 4:  # Friday: closed from 17:00
        return local.time() < FX_WEEK_ROLL
    return True


def tag_fx_sessions(bars: pl.DataFrame) -> pl.DataFrame:
    """Add ``session_date`` plus Asian/London-open window flags to FX bars.

    Columns added: ``session_date`` (London local date), ``in_asian_window``,
    ``in_london_open_window`` (first 15 min after the open), ``is_post_open``
    (at or after the London open).
    """
    london_ts = pl.col("timestamp").dt.convert_time_zone(_LONDON)
    session_date = london_ts.dt.date()
    minutes_of_day = london_ts.dt.hour().cast(pl.Int32) * 60 + london_ts.dt.minute().cast(pl.Int32)
    open_minutes = LONDON_OPEN.hour * 60 + LONDON_OPEN.minute
    return bars.with_columns(
        session_date=session_date,
        in_asian_window=minutes_of_day < open_minutes,
        in_london_open_window=(minutes_of_day >= open_minutes)
        & (minutes_of_day < open_minutes + LONDON_OPEN_FIRST_MINUTES),
        is_post_open=minutes_of_day >= open_minutes,
    )

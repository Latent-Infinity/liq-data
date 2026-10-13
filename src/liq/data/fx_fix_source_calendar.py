"""Candidate FX-fix clock metadata, not source admission or transport authority."""

from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Final
from zoneinfo import ZoneInfo

_LONDON: Final = ZoneInfo("Europe/London")


@dataclass(frozen=True, slots=True)
class FixSourceDay:
    """One weekday's candidate UTC bounds and anchor-ordered native candle marks.

    Marks are pre-start, entry and post-end for 14:00, 15:00 and 16:00 London,
    respectively. Month-end means the last Monday–Friday calendar date, with
    no holiday adjustment. None of these fields establishes available prices.
    """

    day: date
    start_utc: datetime
    end_utc: datetime
    native_marks_utc: tuple[datetime, ...]
    is_month_end: bool


def fx_fix_source_day(day: date) -> FixSourceDay | None:
    """Derive candidate metadata from a calendar date; weekends have no candidate.

    A native M1 candle at U−1 minute supplies the close at event boundary U.
    The range includes every candidate minute between the first and last mark;
    it is not a claim that any candle exists or is complete.
    """
    if day.weekday() >= 5:
        return None
    marks = tuple(
        (datetime.combine(day, time(hour), _LONDON) + timedelta(minutes=offset)).astimezone(UTC)
        for hour in (14, 15, 16)
        for offset in (-21, -1, 29)
    )
    last = date(day.year, day.month, monthrange(day.year, day.month)[1])
    last_weekday = last - timedelta(days=max(last.weekday() - 4, 0))
    return FixSourceDay(
        day=day,
        start_utc=marks[0],
        end_utc=marks[-1] + timedelta(minutes=1, microseconds=-1),
        native_marks_utc=marks,
        is_month_end=day == last_weekday,
    )


def fx_fix_source_year(year: int) -> tuple[FixSourceDay, ...]:
    """Enumerate all candidate weekdays chronologically, including holidays.

    Year selection and permitted source use belong to the caller's reviewed
    inventory. This pure calendar function never accesses a provider or store.
    """
    return tuple(
        candidate
        for month in range(1, 13)
        for day_number in range(1, monthrange(year, month)[1] + 1)
        if (candidate := fx_fix_source_day(date(year, month, day_number))) is not None
    )

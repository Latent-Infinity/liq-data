"""Exchange-clock normalization for SEC filing event metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from liq.data.calendar import _calendar

EventSessionBucket = Literal["in_hours", "before_open", "after_close", "non_session"]

_EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class EdgarEventClock:
    """Tradable clock derived from one public SEC acceptance timestamp."""

    public_time: datetime
    event_time: datetime
    session_date: date
    session_bucket: EventSessionBucket
    session_open: datetime
    session_close: datetime
    reaction_end: datetime
    decision_times: tuple[tuple[int, datetime], ...]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("public_time must be timezone-aware")
    return value.astimezone(UTC)


def _session_open_close(
    session_label: date | str | int | float,
) -> tuple[datetime, datetime]:
    calendar = _calendar()
    session_open = calendar.session_open(session_label).to_pydatetime().astimezone(UTC)
    session_close = calendar.session_close(session_label).to_pydatetime().astimezone(UTC)
    return session_open, session_close


def resolve_edgar_event_clock(
    public_time: datetime,
    *,
    reaction_minutes: int = 5,
    latency_minutes: tuple[int, ...] = (0, 2, 5),
) -> EdgarEventClock:
    """Resolve an SEC acceptance time to the applicable regular session.

    In-hours events retain their public time. Before-open, after-close,
    weekend, and holiday events are re-anchored to the next regular XNYS open.
    """
    if reaction_minutes <= 0:
        raise ValueError("reaction_minutes must be positive")
    if not latency_minutes or any(value < 0 for value in latency_minutes):
        raise ValueError("latency_minutes must be a non-empty tuple of non-negative values")
    if len(set(latency_minutes)) != len(latency_minutes):
        raise ValueError("latency_minutes must not contain duplicates")

    public_utc = _as_utc(public_time)
    local_date = public_utc.astimezone(_EASTERN).date()
    calendar = _calendar()

    if calendar.is_session(local_date.isoformat()):
        current_label = calendar.date_to_session(local_date.isoformat(), direction="none")
        session_open, session_close = _session_open_close(current_label)
        if public_utc < session_open:
            bucket: EventSessionBucket = "before_open"
            event_time = session_open
            session_label = current_label
        elif public_utc < session_close:
            bucket = "in_hours"
            event_time = public_utc
            session_label = current_label
        else:
            bucket = "after_close"
            session_label = calendar.next_session(current_label)
            session_open, session_close = _session_open_close(session_label)
            event_time = session_open
    else:
        bucket = "non_session"
        session_label = calendar.date_to_session(local_date.isoformat(), direction="next")
        session_open, session_close = _session_open_close(session_label)
        event_time = session_open

    reaction_end = event_time + timedelta(minutes=reaction_minutes)
    decisions = tuple(
        (latency, reaction_end + timedelta(minutes=latency)) for latency in latency_minutes
    )
    return EdgarEventClock(
        public_time=public_utc,
        event_time=event_time,
        session_date=session_label.to_pydatetime().date(),
        session_bucket=bucket,
        session_open=session_open,
        session_close=session_close,
        reaction_end=reaction_end,
        decision_times=decisions,
    )

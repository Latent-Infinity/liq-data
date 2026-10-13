"""Current-provider schedule mapping for the fixed retrospective SEC corpus."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Final, Never

from pydantic import TypeAdapter, ValidationError

from liq.data.alpaca_announcement_source import (
    ANNOUNCEMENT_DATE_PAIRS,
    AlpacaAnnouncementCalendarRequest,
    AnnouncementAccession,
)
from liq.data.alpaca_calendar import (
    CalendarReference,
    CalendarRefused,
    CalendarSession,
    _clock,
    _Row,
    _unique_keys,
)

# Accepted documentary clocks, not earliest press-release publication claims.
_EVENT_CLOCKS: Final[tuple[tuple[AnnouncementAccession, datetime], ...]] = (
    ("0001637459-22-000056", datetime(2022, 4, 27, 11, 1, 58, tzinfo=UTC)),
    ("0001637459-22-000156", datetime(2022, 7, 27, 11, 9, 12, tzinfo=UTC)),
    ("0001637459-22-000186", datetime(2022, 10, 26, 11, 1, 34, tzinfo=UTC)),
    ("0001637459-23-000005", datetime(2023, 2, 15, 12, 1, 44, tzinfo=UTC)),
)


@dataclass(frozen=True, slots=True)
class AnnouncementSessionPair:
    accession: AnnouncementAccession
    acceptance_utc: datetime
    preceding: CalendarSession
    following: CalendarSession


@dataclass(frozen=True, slots=True)
class AnnouncementCalendar:
    """Returned schedule only; no assertion of venue coverage or omitted holidays."""

    reference: CalendarReference
    event_pairs: tuple[AnnouncementSessionPair, ...]


def _refuse_constant(_label: str) -> Never:
    raise CalendarRefused("announcement_calendar_nonfinite_json")


def parse_announcement_calendar(raw: bytes, captured_at: datetime) -> AnnouncementCalendar:
    """Refuse disagreement; never sort, fill or substitute a nearby session.

    This parser supplies neither source-access permission nor financial acceptance.
    Provider-declared US calendar metadata does not independently establish XNAS/
    ARCX coverage, historical availability, corrections, actions or price semantics.
    """
    request = AlpacaAnnouncementCalendarRequest()
    if len(raw) > request.max_response_bytes:
        raise CalendarRefused("announcement_calendar_byte_budget")
    if captured_at.utcoffset() != timedelta(0) or captured_at.date() < request.end:
        raise CalendarRefused("announcement_calendar_capture_clock")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_refuse_constant)
        rows = TypeAdapter(list[_Row]).validate_python(payload, strict=True)
    except CalendarRefused:
        raise
    except (ValueError, UnicodeDecodeError, ValidationError) as exc:
        raise CalendarRefused("announcement_calendar_schema") from exc
    if not rows or len(rows) > request.max_records:
        raise CalendarRefused("announcement_calendar_record_budget")
    sessions: list[CalendarSession] = []
    for row in rows:
        try:
            day = date.fromisoformat(row.date)
        except ValueError as exc:
            raise CalendarRefused("announcement_calendar_date") from exc
        if day.isoformat() != row.date:
            raise CalendarRefused("announcement_calendar_date_label")
        opened, closed = _clock(day, row.open), _clock(day, row.close)
        if opened >= closed or closed > captured_at:
            raise CalendarRefused("announcement_calendar_session_interval")
        sessions.append(CalendarSession(day, row.open, row.close, opened, closed))
    dates = tuple(session.date for session in sessions)
    if (
        dates[0] != request.start
        or dates[-1] != request.end
        or any(day.weekday() >= 5 for day in dates)
        or any(left >= right for left, right in pairwise(dates))
    ):
        raise CalendarRefused("announcement_calendar_date_scope")
    pairs: list[AnnouncementSessionPair] = []
    for accession, clock in _EVENT_CLOCKS:
        preceding = next(
            (session for session in reversed(sessions) if session.close_utc < clock), None
        )
        following = next((session for session in sessions if session.open_utc > clock), None)
        if preceding is None or following is None:
            raise CalendarRefused("announcement_calendar_event_boundary_missing")
        if (preceding.date, following.date) != ANNOUNCEMENT_DATE_PAIRS[accession]:
            raise CalendarRefused("announcement_calendar_event_pair_disagrees")
        pairs.append(AnnouncementSessionPair(accession, clock, preceding, following))
    reference = CalendarReference(tuple(sessions), hashlib.sha256(raw).hexdigest(), captured_at)
    return AnnouncementCalendar(reference, tuple(pairs))

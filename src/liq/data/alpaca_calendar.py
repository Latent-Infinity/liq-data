"""Strict current-provider schedule metadata; never historical price/fill authority."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, StrictStr, TypeAdapter, ValidationError

from liq.data.alpaca_source import AlpacaCalendarRequest


class CalendarRefused(ValueError):
    """Malformed or disagreeing reference schedule; never repaired."""


class _Row(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="ignore")
    date: StrictStr
    open: StrictStr
    close: StrictStr


@dataclass(frozen=True, slots=True)
class CalendarSession:
    date: date
    original_open: str
    original_close: str
    open_utc: datetime
    close_utc: datetime


@dataclass(frozen=True, slots=True)
class CalendarReference:
    sessions: tuple[CalendarSession, ...]
    source_sha256: str
    captured_at: datetime
    timezone: str = "America/New_York"


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CalendarRefused("duplicate_json_key")
        result[key] = value
    return result


def _clock(day: date, label: str) -> datetime:
    if not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", label):
        raise CalendarRefused("invalid_clock")
    try:
        local = datetime.combine(day, datetime.strptime(label, "%H:%M").time())
    except ValueError as exc:
        raise CalendarRefused("invalid_clock") from exc
    eastern = ZoneInfo("America/New_York")
    first = local.replace(tzinfo=eastern, fold=0)
    second = local.replace(tzinfo=eastern, fold=1)
    if (
        first.utcoffset() != second.utcoffset()
        or first.astimezone(UTC).astimezone(eastern).replace(tzinfo=None) != local
    ):
        raise CalendarRefused("ambiguous_or_nonexistent_clock")
    return first.astimezone(UTC)


def parse_calendar(
    raw: bytes, captured_at: datetime, expected_dates: tuple[date, ...]
) -> CalendarReference:
    """Decode only retained bytes, exact predeclared labels and explicit US Eastern clocks.

    The caller retains the raw response first. Missing, reordered or duplicate
    labels refuse interpretation. Calendar agreement conveys no price/auction,
    membership, corporate-action or original-publication admission.
    """
    request = AlpacaCalendarRequest()
    if len(raw) > request.max_response_bytes:
        raise CalendarRefused("byte_budget")
    if captured_at.utcoffset() != timedelta(0) or captured_at.date() < request.end:
        raise CalendarRefused("invalid_capture_clock")
    if (
        not expected_dates
        or len(expected_dates) > 90
        or any(
            type(day) is not date or not request.start <= day <= request.end
            for day in expected_dates
        )
        or any(a >= b for a, b in pairwise(expected_dates))
    ):
        raise CalendarRefused("invalid_expected_dates")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_keys)
        rows = TypeAdapter(list[_Row]).validate_python(payload, strict=True)
    except CalendarRefused:
        raise
    except (ValueError, UnicodeDecodeError, ValidationError) as exc:
        raise CalendarRefused("invalid_schema") from exc
    if not rows or len(rows) > 90:
        raise CalendarRefused("record_budget")
    try:
        dates = tuple(date.fromisoformat(row.date) for row in rows)
    except ValueError as exc:
        raise CalendarRefused("invalid_date") from exc
    if dates != expected_dates or any(
        day.isoformat() != row.date for day, row in zip(dates, rows, strict=True)
    ):
        raise CalendarRefused("date_labels_disagree")
    sessions = []
    for day, row in zip(dates, rows, strict=True):
        opened, closed = _clock(day, row.open), _clock(day, row.close)
        if opened >= closed:
            raise CalendarRefused("invalid_session_interval")
        sessions.append(CalendarSession(day, row.open, row.close, opened, closed))
    return CalendarReference(tuple(sessions), hashlib.sha256(raw).hexdigest(), captured_at)

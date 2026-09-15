"""Calendar-only DEV_SMOKE_ONLY checks; no market observations."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from liq.data.session_clock import regular_session_minutes, shift_session_minute


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


@pytest.mark.parametrize(
    ("timestamp", "sessions", "expected"),
    [
        ("2024-03-11T13:30", -1, "2024-03-08T14:30"),
        ("2024-03-08T14:30", 1, "2024-03-11T13:30"),
        ("2024-11-04T14:30", -1, "2024-11-01T13:30"),
        ("2024-07-05T13:30", -1, "2024-07-03T13:30"),
        ("2024-01-02T14:30", -1, "2023-12-29T14:30"),
        ("2024-03-11T19:59", 0, "2024-03-11T19:59"),
        ("2024-03-12T13:30", -2, "2024-03-08T14:30"),
        ("2003-01-02T14:30", 1, "2003-01-03T14:30"),
    ],
)
def test_shift_preserves_exchange_clock(timestamp: str, sessions: int, expected: str) -> None:
    given = utc(timestamp)
    actual = shift_session_minute(given, sessions)
    assert actual == utc(expected)


def test_shift_returns_none_when_target_early_close_has_no_matching_minute() -> None:
    given = utc("2024-07-05T17:00")
    actual = shift_session_minute(given, -1)
    assert actual is None


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2024, 3, 11, 13, 30),
        utc("2024-03-11T13:30:01"),
        utc("2024-03-11T13:30:00.000001"),
        utc("2024-03-11T13:29"),
        utc("2024-03-11T20:00"),
        utc("2024-03-10T13:30"),
        utc("2024-07-04T13:30"),
        utc("2024-03-11T13:30").astimezone(ZoneInfo("America/New_York")),
    ],
)
def test_shift_rejects_invalid_bar_start(timestamp: datetime) -> None:
    with pytest.raises(ValueError):
        shift_session_minute(timestamp, 0)


def test_grid_is_half_open_and_omits_closed_sessions() -> None:
    start, end = utc("2024-07-03T16:58"), utc("2024-07-05T13:32")
    actual = regular_session_minutes(start, end)
    assert actual == tuple(
        utc(value)
        for value in (
            "2024-07-03T16:58",
            "2024-07-03T16:59",
            "2024-07-05T13:30",
            "2024-07-05T13:31",
        )
    )


def test_grid_has_390_regular_bar_starts() -> None:
    start = utc("2024-03-11T13:30")
    actual = regular_session_minutes(start, utc("2024-03-11T20:00"))
    assert actual == tuple(start + timedelta(minutes=i) for i in range(390))


def test_empty_grid_when_end_equals_start() -> None:
    instant = utc("2024-03-11T13:30")
    actual = regular_session_minutes(instant, instant)
    assert actual == ()


def test_grid_rejects_reversed_bounds() -> None:
    with pytest.raises(ValueError):
        regular_session_minutes(utc("2024-03-12T13:30"), utc("2024-03-11T13:30"))


@pytest.mark.parametrize("sessions", [True, 1.5])
def test_shift_rejects_non_integer_offset(sessions: int | float) -> None:
    with pytest.raises(ValueError):
        shift_session_minute(utc("2024-03-11T13:30"), sessions)

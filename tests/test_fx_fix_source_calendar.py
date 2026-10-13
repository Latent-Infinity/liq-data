"""Calendar-only contracts: no fabricated market observations or transport."""

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta

import pytest

from liq.data.fx_fix_source_calendar import fx_fix_source_day, fx_fix_source_year


@pytest.mark.parametrize(
    ("day", "utc_hour"),
    [
        (date(2024, 1, 31), 13),
        (date(2024, 3, 28), 13),
        (date(2024, 4, 2), 12),
        (date(2024, 7, 3), 12),
        (date(2024, 10, 25), 12),
        (date(2024, 10, 28), 13),
    ],
)
def test_window_tracks_london_dst_when_candidate_day_changes(day: date, utc_hour: int) -> None:
    result = fx_fix_source_day(day)

    assert result is not None
    assert result.start_utc == datetime(day.year, day.month, day.day, utc_hour, 39, tzinfo=UTC)
    assert result.end_utc == datetime(
        day.year, day.month, day.day, utc_hour + 3, 29, 59, 999999, tzinfo=UTC
    )
    assert result.end_utc - result.start_utc == timedelta(minutes=171, microseconds=-1)


def test_native_marks_use_preceding_complete_candle_when_anchors_are_declared() -> None:
    result = fx_fix_source_day(date(2024, 7, 3))

    assert result is not None
    assert result.native_marks_utc == tuple(
        datetime(2024, 7, 3, hour, minute, tzinfo=UTC)
        for hour, minute in (
            (12, 39),
            (12, 59),
            (13, 29),
            (13, 39),
            (13, 59),
            (14, 29),
            (14, 39),
            (14, 59),
            (15, 29),
        )
    )


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2024, 3, 28), False),
        (date(2024, 3, 29), True),
        (date(2024, 8, 30), True),
        (date(2024, 8, 29), False),
        (date(2024, 2, 29), True),
        (date(2023, 2, 28), True),
        (date(2024, 10, 31), True),
    ],
)
def test_month_end_is_last_calendar_weekday_not_holiday_adjustment(
    day: date, expected: bool
) -> None:
    result = fx_fix_source_day(day)

    assert result is not None
    assert result.is_month_end is expected


@pytest.mark.parametrize(
    ("year", "expected"), [(2020, 262), (2021, 261), (2022, 260), (2023, 260), (2024, 262)]
)
def test_year_enumerates_all_weekdays_without_holiday_filtering(year: int, expected: int) -> None:
    result = fx_fix_source_year(year)

    assert len(result) == expected
    assert len({item.day for item in result}) == expected
    assert all(item.day.year == year and item.day.weekday() < 5 for item in result)
    assert tuple(item.day for item in result) == tuple(sorted(item.day for item in result))
    assert sum(item.is_month_end for item in result) == 12


def test_leap_day_is_retained_when_year_is_leap() -> None:
    result = fx_fix_source_year(2024)

    assert date(2024, 2, 29) in {item.day for item in result}
    assert date(2024, 3, 29) in {item.day for item in result}  # Good Friday is a weekday.


@pytest.mark.parametrize("day", [date(2024, 3, 30), date(2024, 3, 31)])
def test_weekend_has_no_candidate_when_calendar_day_is_requested(day: date) -> None:
    assert fx_fix_source_day(day) is None


@pytest.mark.parametrize("field_name", ["day", "end_utc"])
def test_day_metadata_is_immutable_when_field_replacement_is_attempted(field_name: str) -> None:
    result = fx_fix_source_day(date(2024, 1, 31))
    assert result is not None

    with pytest.raises(FrozenInstanceError):
        setattr(result, field_name, None)

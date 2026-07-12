"""Tests for FX session tagging (Asian range, London open, FX week)."""

from datetime import UTC, date, datetime

import polars as pl
import pytest

from liq.data.fx_session import (
    asian_range_window_utc,
    fx_session_date,
    is_fx_trading_week,
    london_open_utc,
    london_open_window_utc,
    tag_fx_sessions,
)


class TestLondonOpenUtc:
    def test_winter_gmt(self) -> None:
        # January: GMT, 08:00 London == 08:00 UTC.
        assert london_open_utc(date(2023, 1, 16)) == datetime(2023, 1, 16, 8, 0, tzinfo=UTC)

    def test_summer_bst(self) -> None:
        # June: BST (UTC+1), 08:00 London == 07:00 UTC.
        assert london_open_utc(date(2023, 6, 15)) == datetime(2023, 6, 15, 7, 0, tzinfo=UTC)

    def test_spring_forward_day(self) -> None:
        # 2023-03-26 clocks jump 01:00 GMT -> 02:00 BST; 08:00 is BST == 07:00 UTC.
        assert london_open_utc(date(2023, 3, 26)) == datetime(2023, 3, 26, 7, 0, tzinfo=UTC)

    def test_fall_back_day(self) -> None:
        # 2023-10-29 clocks fall 02:00 BST -> 01:00 GMT; 08:00 is GMT == 08:00 UTC.
        assert london_open_utc(date(2023, 10, 29)) == datetime(2023, 10, 29, 8, 0, tzinfo=UTC)


class TestAsianRangeWindow:
    def test_winter_window(self) -> None:
        start, end = asian_range_window_utc(date(2023, 1, 16))
        assert start == datetime(2023, 1, 16, 0, 0, tzinfo=UTC)
        assert end == datetime(2023, 1, 16, 8, 0, tzinfo=UTC)

    def test_summer_window(self) -> None:
        start, end = asian_range_window_utc(date(2023, 6, 15))
        # 00:00 BST == 2023-06-14 23:00 UTC; 08:00 BST == 07:00 UTC.
        assert start == datetime(2023, 6, 14, 23, 0, tzinfo=UTC)
        assert end == datetime(2023, 6, 15, 7, 0, tzinfo=UTC)

    def test_spring_forward_window_is_seven_utc_hours(self) -> None:
        # The 01:00-02:00 gap means 00:00-08:00 London spans only 7 UTC hours.
        start, end = asian_range_window_utc(date(2023, 3, 26))
        assert start == datetime(2023, 3, 26, 0, 0, tzinfo=UTC)
        assert end == datetime(2023, 3, 26, 7, 0, tzinfo=UTC)
        assert (end - start).total_seconds() == 7 * 3600

    def test_fall_back_window_is_nine_utc_hours(self) -> None:
        # The repeated hour means 00:00-08:00 London spans 9 UTC hours.
        start, end = asian_range_window_utc(date(2023, 10, 29))
        assert start == datetime(2023, 10, 28, 23, 0, tzinfo=UTC)
        assert end == datetime(2023, 10, 29, 8, 0, tzinfo=UTC)
        assert (end - start).total_seconds() == 9 * 3600

    def test_window_end_equals_london_open(self) -> None:
        for day in (date(2023, 1, 16), date(2023, 6, 15)):
            _, end = asian_range_window_utc(day)
            assert end == london_open_utc(day)


class TestLondonOpenWindow:
    def test_first_15_minutes(self) -> None:
        start, end = london_open_window_utc(date(2023, 6, 15), minutes=15)
        assert start == datetime(2023, 6, 15, 7, 0, tzinfo=UTC)
        assert end == datetime(2023, 6, 15, 7, 15, tzinfo=UTC)

    def test_default_is_15(self) -> None:
        start, end = london_open_window_utc(date(2023, 1, 16))
        assert (end - start).total_seconds() == 15 * 60

    def test_invalid_minutes(self) -> None:
        with pytest.raises(ValueError, match="minutes"):
            london_open_window_utc(date(2023, 1, 16), minutes=0)


class TestFxSessionDate:
    def test_london_local_date(self) -> None:
        # 2023-06-14 23:30 UTC == 2023-06-15 00:30 BST -> London date 2023-06-15.
        assert fx_session_date(datetime(2023, 6, 14, 23, 30, tzinfo=UTC)) == date(2023, 6, 15)

    def test_winter_same_date(self) -> None:
        assert fx_session_date(datetime(2023, 1, 16, 3, 0, tzinfo=UTC)) == date(2023, 1, 16)

    def test_requires_tz_aware(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            fx_session_date(datetime(2023, 1, 16, 3, 0))


class TestFxTradingWeek:
    def test_weekday_open(self) -> None:
        assert is_fx_trading_week(datetime(2023, 6, 14, 12, 0, tzinfo=UTC)) is True

    def test_saturday_closed(self) -> None:
        assert is_fx_trading_week(datetime(2023, 6, 17, 12, 0, tzinfo=UTC)) is False

    def test_sunday_before_open_closed(self) -> None:
        # Sunday 2023-06-18 12:00 UTC is before the Sunday 17:00 NY open.
        assert is_fx_trading_week(datetime(2023, 6, 18, 12, 0, tzinfo=UTC)) is False

    def test_sunday_after_open(self) -> None:
        # Sunday 2023-06-18 22:00 UTC == 18:00 EDT, after the 17:00 NY open.
        assert is_fx_trading_week(datetime(2023, 6, 18, 22, 0, tzinfo=UTC)) is True

    def test_friday_after_close(self) -> None:
        # Friday 2023-06-16 22:00 UTC == 18:00 EDT, after the 17:00 NY close.
        assert is_fx_trading_week(datetime(2023, 6, 16, 22, 0, tzinfo=UTC)) is False


class TestTagFxSessions:
    def _bars(self) -> pl.DataFrame:
        # Bars around the 2023-06-15 London session (BST: open 07:00 UTC).
        times = [
            datetime(2023, 6, 15, 2, 0, tzinfo=UTC),  # Asian window
            datetime(2023, 6, 15, 6, 30, tzinfo=UTC),  # Asian window
            datetime(2023, 6, 15, 7, 0, tzinfo=UTC),  # London open (first 15m)
            datetime(2023, 6, 15, 7, 10, tzinfo=UTC),  # first 15m
            datetime(2023, 6, 15, 9, 0, tzinfo=UTC),  # post-open
        ]
        return pl.DataFrame(
            {
                "timestamp": times,
                "open": [1.0] * 5,
                "high": [1.1] * 5,
                "low": [0.9] * 5,
                "close": [1.05] * 5,
                "volume": [10.0] * 5,
            }
        )

    def test_tags_windows(self) -> None:
        tagged = tag_fx_sessions(self._bars())
        assert tagged["session_date"].to_list() == [date(2023, 6, 15)] * 5
        assert tagged["in_asian_window"].to_list() == [True, True, False, False, False]
        assert tagged["in_london_open_window"].to_list() == [
            False,
            False,
            True,
            True,
            False,
        ]

    def test_post_open_flag(self) -> None:
        tagged = tag_fx_sessions(self._bars())
        assert tagged["is_post_open"].to_list() == [
            False,
            False,
            True,
            True,
            True,
        ]

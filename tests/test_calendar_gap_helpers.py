"""Tests for the closed-market gap helpers ``closed_hours_between``
and ``classify_gap`` in ``liq.data.calendar``.

These helpers are consumed by ``liq-features``' volatility decomposition
to label ``overnight_gap_var_t`` with its closed-market context — the raw
empirical gap stays unscaled, but the classification + per-closed-hour
diagnostic feeds regime logic without ever scaling the canonical scalar.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from liq.data.calendar import classify_gap, closed_hours_between


class TestClosedHoursBetween:
    """``closed_hours_between(c_prev, o)`` returns the elapsed duration
    in hours between a previous session close and the next open."""

    def test_weeknight_standard_time_returns_17_5(self) -> None:
        # Wed 2025-01-15 21:00 UTC close (16:00 ET, standard time)
        # → Thu 2025-01-16 14:30 UTC open (09:30 ET).
        c_prev = datetime(2025, 1, 15, 21, 0, tzinfo=UTC)
        o = datetime(2025, 1, 16, 14, 30, tzinfo=UTC)
        assert closed_hours_between(c_prev, o) == pytest.approx(17.5)

    def test_weeknight_dst_returns_17_5(self) -> None:
        # Wed 2024-06-05 20:00 UTC close (16:00 ET, DST)
        # → Thu 2024-06-06 13:30 UTC open (09:30 ET).
        c_prev = datetime(2024, 6, 5, 20, 0, tzinfo=UTC)
        o = datetime(2024, 6, 6, 13, 30, tzinfo=UTC)
        assert closed_hours_between(c_prev, o) == pytest.approx(17.5)

    def test_weekend_no_holiday_returns_65_5(self) -> None:
        # Fri 2024-06-07 20:00 UTC close → Mon 2024-06-10 13:30 UTC open.
        c_prev = datetime(2024, 6, 7, 20, 0, tzinfo=UTC)
        o = datetime(2024, 6, 10, 13, 30, tzinfo=UTC)
        assert closed_hours_between(c_prev, o) == pytest.approx(65.5)

    def test_weekend_spanning_dst_spring_forward_returns_64_5(self) -> None:
        # DST 2024 began Sun 2024-03-10. Fri 21:00 UTC (EST) → Mon 13:30
        # UTC (EDT) — UTC-aware subtraction gives elapsed
        # hours are exactly 64.5; the wall-clock "spring forward" is
        # absorbed by switching from 16:00 ET close to 09:30 ET open.
        c_prev = datetime(2024, 3, 8, 21, 0, tzinfo=UTC)
        o = datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
        assert closed_hours_between(c_prev, o) == pytest.approx(64.5)

    def test_three_day_weekend_mlk_returns_89_5(self) -> None:
        # MLK Day 2025: Mon 2025-01-20 is a NYSE holiday.
        # Fri 2025-01-17 21:00 UTC → Tue 2025-01-21 14:30 UTC = 89.5h.
        c_prev = datetime(2025, 1, 17, 21, 0, tzinfo=UTC)
        o = datetime(2025, 1, 21, 14, 30, tzinfo=UTC)
        assert closed_hours_between(c_prev, o) == pytest.approx(89.5)

    def test_halt_reopen_intraday_returns_fractional_hours(self) -> None:
        # Same-day halt: a 90-minute halt at 17:00–18:30 UTC.
        c_prev = datetime(2024, 6, 3, 17, 0, tzinfo=UTC)
        o = datetime(2024, 6, 3, 18, 30, tzinfo=UTC)
        assert closed_hours_between(c_prev, o) == pytest.approx(1.5)

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            closed_hours_between(
                datetime(2024, 6, 3, 21, 0), datetime(2024, 6, 4, 13, 30, tzinfo=UTC)
            )
        with pytest.raises(ValueError, match="tz-aware"):
            closed_hours_between(
                datetime(2024, 6, 3, 21, 0, tzinfo=UTC), datetime(2024, 6, 4, 13, 30)
            )

    def test_open_before_close_raises(self) -> None:
        with pytest.raises(ValueError, match="at or after"):
            closed_hours_between(
                datetime(2024, 6, 4, 13, 30, tzinfo=UTC),
                datetime(2024, 6, 3, 21, 0, tzinfo=UTC),
            )


class TestClassifyGap:
    """``classify_gap(c_prev, o)`` labels the closed-market interval
    with one of the categories from research plan §3.1a."""

    def test_weeknight_adjacent_sessions(self) -> None:
        c_prev = datetime(2024, 6, 5, 20, 0, tzinfo=UTC)  # Wed close
        o = datetime(2024, 6, 6, 13, 30, tzinfo=UTC)  # Thu open
        assert classify_gap(c_prev, o) == "weeknight"

    def test_weekend_friday_to_monday(self) -> None:
        c_prev = datetime(2024, 6, 7, 20, 0, tzinfo=UTC)  # Fri close
        o = datetime(2024, 6, 10, 13, 30, tzinfo=UTC)  # Mon open
        assert classify_gap(c_prev, o) == "weekend"

    def test_pre_holiday_isolated_midweek(self) -> None:
        # July 4 2024 is a Thursday NYSE holiday.
        # Wed 2024-07-03 17:00 UTC early-close → Fri 2024-07-05 13:30 UTC.
        # The day immediately after Wed (Thu) is the holiday → pre_holiday.
        c_prev = datetime(2024, 7, 3, 17, 0, tzinfo=UTC)
        o = datetime(2024, 7, 5, 13, 30, tzinfo=UTC)
        assert classify_gap(c_prev, o) == "pre_holiday"

    def test_long_holiday_three_day_weekend_mlk(self) -> None:
        # MLK Day 2025: Mon 2025-01-20 is a holiday → Fri→Tue gap with
        # weekend + holiday interior.
        c_prev = datetime(2025, 1, 17, 21, 0, tzinfo=UTC)
        o = datetime(2025, 1, 21, 14, 30, tzinfo=UTC)
        assert classify_gap(c_prev, o) == "long_holiday"

    def test_long_holiday_thanksgiving_thursday(self) -> None:
        # Thanksgiving 2024 Thu 2024-11-28; Fri Black Friday is open (early
        # close). Wed 2024-11-27 close → Fri 2024-11-29 open spans Thu
        # holiday only → pre_holiday (single-day mid-week holiday).
        c_prev = datetime(2024, 11, 27, 21, 0, tzinfo=UTC)
        o = datetime(2024, 11, 29, 14, 30, tzinfo=UTC)
        assert classify_gap(c_prev, o) == "pre_holiday"

    def test_long_holiday_good_friday_weekend(self) -> None:
        # Good Friday 2024-03-29 is closed; Thu close → Mon open spans
        # Fri (holiday) + Sat + Sun → weekend + holiday → long_holiday.
        c_prev = datetime(2024, 3, 28, 20, 0, tzinfo=UTC)
        o = datetime(2024, 4, 1, 13, 30, tzinfo=UTC)
        assert classify_gap(c_prev, o) == "long_holiday"

    def test_halt_reopen_same_session(self) -> None:
        # Intraday halt and resume within the same XNYS session.
        c_prev = datetime(2024, 6, 3, 17, 0, tzinfo=UTC)
        o = datetime(2024, 6, 3, 18, 30, tzinfo=UTC)
        assert classify_gap(c_prev, o) == "halt_reopen"

    def test_dst_weekend_classifies_as_weekend(self) -> None:
        # Fri 2024-03-08 → Mon 2024-03-11 spans DST start but no holiday.
        c_prev = datetime(2024, 3, 8, 21, 0, tzinfo=UTC)
        o = datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
        assert classify_gap(c_prev, o) == "weekend"

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            classify_gap(datetime(2024, 6, 3, 20, 0), datetime(2024, 6, 4, 13, 30, tzinfo=UTC))

    def test_open_before_close_raises(self) -> None:
        with pytest.raises(ValueError, match="at or after"):
            classify_gap(
                datetime(2024, 6, 6, 13, 30, tzinfo=UTC),
                datetime(2024, 6, 5, 20, 0, tzinfo=UTC),
            )

"""Clock-boundary tests for SEC filing event metadata."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from liq.data.edgar_events import resolve_edgar_event_clock


class TestResolveEdgarEventClock:
    def test_in_hours_event_preserves_public_time(self) -> None:
        clock = resolve_edgar_event_clock(
            datetime(2023, 8, 3, 16, 30, 15, tzinfo=UTC),
            reaction_minutes=5,
            latency_minutes=(0, 2, 5),
        )

        assert clock.session_bucket == "in_hours"
        assert clock.session_open == datetime(2023, 8, 3, 13, 30, tzinfo=UTC)
        assert clock.session_close == datetime(2023, 8, 3, 20, 0, tzinfo=UTC)
        assert clock.event_time == datetime(2023, 8, 3, 16, 30, 15, tzinfo=UTC)
        assert clock.reaction_end == datetime(2023, 8, 3, 16, 35, 15, tzinfo=UTC)
        assert clock.decision_times == (
            (0, datetime(2023, 8, 3, 16, 35, 15, tzinfo=UTC)),
            (2, datetime(2023, 8, 3, 16, 37, 15, tzinfo=UTC)),
            (5, datetime(2023, 8, 3, 16, 40, 15, tzinfo=UTC)),
        )

    def test_after_close_event_rebuckets_to_next_regular_open(self) -> None:
        clock = resolve_edgar_event_clock(
            datetime(2023, 8, 3, 21, 30, tzinfo=UTC),
        )

        assert clock.session_bucket == "after_close"
        assert clock.session_date.isoformat() == "2023-08-04"
        assert clock.session_open == datetime(2023, 8, 4, 13, 30, tzinfo=UTC)
        assert clock.session_close == datetime(2023, 8, 4, 20, 0, tzinfo=UTC)
        assert clock.event_time == datetime(2023, 8, 4, 13, 30, tzinfo=UTC)

    def test_before_open_event_rebuckets_to_same_day_open(self) -> None:
        clock = resolve_edgar_event_clock(
            datetime(2023, 8, 3, 12, 0, tzinfo=UTC),
        )

        assert clock.session_bucket == "before_open"
        assert clock.event_time == datetime(2023, 8, 3, 13, 30, tzinfo=UTC)

    def test_weekend_event_rebuckets_to_monday_open(self) -> None:
        clock = resolve_edgar_event_clock(
            datetime(2023, 8, 5, 16, 0, tzinfo=UTC),
        )

        assert clock.session_bucket == "non_session"
        assert clock.session_date.isoformat() == "2023-08-07"
        assert clock.event_time == datetime(2023, 8, 7, 13, 30, tzinfo=UTC)

    def test_early_close_boundary_uses_calendar_close(self) -> None:
        clock = resolve_edgar_event_clock(
            datetime(2024, 11, 29, 18, 30, tzinfo=UTC),
        )

        assert clock.session_bucket == "after_close"
        assert clock.event_time == datetime(2024, 12, 2, 14, 30, tzinfo=UTC)

    def test_rejects_naive_timestamp_and_invalid_parameters(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            resolve_edgar_event_clock(datetime(2023, 8, 3, 16, 30))
        with pytest.raises(ValueError, match="reaction_minutes"):
            resolve_edgar_event_clock(
                datetime(2023, 8, 3, 16, 30, tzinfo=UTC),
                reaction_minutes=0,
            )
        with pytest.raises(ValueError, match="latency_minutes"):
            resolve_edgar_event_clock(
                datetime(2023, 8, 3, 16, 30, tzinfo=UTC),
                latency_minutes=(0, -1),
            )

"""Tests for ``liq.data.calendar`` — trading-session helpers used by
the scanner to translate window specs into concrete UTC bounds.

The helpers wrap ``exchange_calendars.XNYS`` so the rest of the LIQ
stack does not need to know about ``exchange_calendars`` directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from liq.data.calendar import (
    nyse_session_close,
    trading_minutes_window,
    trading_sessions_window,
)


class TestTradingMinutesWindow:
    """``trading_minutes_window(end, n)`` returns the ``(start, end)``
    half-open window covering the previous ``n`` trading minutes."""

    def test_full_session_window_of_390_minutes(self) -> None:
        # NYSE regular session is 09:30-16:00 ET → 13:30-20:00 UTC (DST).
        # 2024-06-03 is a regular Monday.
        end = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
        start, resolved_end = trading_minutes_window(end, 390)
        assert resolved_end == end
        assert start == datetime(2024, 6, 3, 13, 30, tzinfo=UTC)

    def test_small_window_aligns_to_minute_grain(self) -> None:
        end = datetime(2024, 6, 3, 14, 0, tzinfo=UTC)
        start, _ = trading_minutes_window(end, 30)
        assert start == datetime(2024, 6, 3, 13, 30, tzinfo=UTC)

    def test_friday_to_monday_spanner(self) -> None:
        # Asking for 30 trading minutes ending Monday 13:35 UTC should
        # span Friday's close into Monday's open.
        end = datetime(2024, 6, 3, 13, 35, tzinfo=UTC)
        start, _ = trading_minutes_window(end, 30)
        # 5 minutes from Monday + 25 from Friday's close (last 25 minutes
        # of the regular session).
        assert start == datetime(2024, 5, 31, 19, 35, tzinfo=UTC)

    def test_rejects_zero_or_negative_n(self) -> None:
        end = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="positive"):
            trading_minutes_window(end, 0)
        with pytest.raises(ValueError, match="positive"):
            trading_minutes_window(end, -1)


class TestTradingSessionsWindow:
    """``trading_sessions_window(end, n)`` returns the open-of-session-N
    through ``end`` window — useful for "last N trading days"."""

    def test_one_session_returns_open_of_same_session(self) -> None:
        end = datetime(2024, 6, 3, 20, 0, tzinfo=UTC)
        start, resolved_end = trading_sessions_window(end, 1)
        assert start == datetime(2024, 6, 3, 13, 30, tzinfo=UTC)
        assert resolved_end == end

    def test_three_sessions_walks_back_two_sessions(self) -> None:
        end = datetime(2024, 6, 5, 20, 0, tzinfo=UTC)
        start, _ = trading_sessions_window(end, 3)
        # Mon, Tue, Wed → start = Mon open
        assert start == datetime(2024, 6, 3, 13, 30, tzinfo=UTC)


class TestNyseSessionClose:
    def test_regular_session_close_is_2000_utc(self) -> None:
        close = nyse_session_close(datetime(2024, 6, 3, tzinfo=UTC).date())
        assert close == datetime(2024, 6, 3, 20, 0, tzinfo=UTC)

    def test_early_close_session_is_1800_utc(self) -> None:
        # 2024-11-29 (day after Thanksgiving) closes at 13:00 ET = 18:00 UTC.
        close = nyse_session_close(datetime(2024, 11, 29, tzinfo=UTC).date())
        assert close == datetime(2024, 11, 29, 18, 0, tzinfo=UTC)

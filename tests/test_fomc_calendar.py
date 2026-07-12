"""Tests for the FOMC scheduled-announcement calendar."""

from datetime import UTC, date, datetime

from liq.data.fomc_calendar import (
    FomcAnnouncement,
    load_fomc_announcements,
)


class TestLoadFomcAnnouncements:
    def test_full_range_count(self) -> None:
        """63 scheduled announcements 2015-2022 (2020 has 7)."""
        events = load_fomc_announcements(date(2015, 1, 1), date(2022, 12, 31))
        assert len(events) == 63

    def test_2020_has_seven_scheduled(self) -> None:
        events = load_fomc_announcements(date(2020, 1, 1), date(2020, 12, 31))
        assert len(events) == 7
        dates = [e.announcement_date for e in events]
        assert date(2020, 3, 15) not in dates
        assert date(2020, 3, 3) not in dates

    def test_every_other_year_has_eight(self) -> None:
        for year in (2015, 2016, 2017, 2018, 2019, 2021, 2022):
            events = load_fomc_announcements(date(year, 1, 1), date(year, 12, 31))
            assert len(events) == 8, year

    def test_spot_dates(self) -> None:
        events = load_fomc_announcements(date(2015, 1, 1), date(2022, 12, 31))
        dates = {e.announcement_date for e in events}
        assert date(2015, 1, 28) in dates
        assert date(2017, 2, 1) in dates
        assert date(2018, 8, 1) in dates
        assert date(2019, 10, 4) not in dates  # unscheduled call excluded
        assert date(2022, 12, 14) in dates

    def test_announcement_time_is_1400_et(self) -> None:
        events = load_fomc_announcements(date(2022, 1, 1), date(2022, 12, 31))
        assert all(e.announcement_time_et == "14:00" for e in events)

    def test_utc_conversion_handles_dst(self) -> None:
        events = load_fomc_announcements(date(2022, 1, 1), date(2022, 12, 31))
        by_date = {e.announcement_date: e for e in events}
        # January (EST): 14:00 ET = 19:00 UTC
        jan = by_date[date(2022, 1, 26)]
        assert jan.announcement_utc == datetime(2022, 1, 26, 19, 0, tzinfo=UTC)
        # June (EDT): 14:00 ET = 18:00 UTC
        jun = by_date[date(2022, 6, 15)]
        assert jun.announcement_utc == datetime(2022, 6, 15, 18, 0, tzinfo=UTC)

    def test_meeting_start_precedes_announcement(self) -> None:
        events = load_fomc_announcements(date(2015, 1, 1), date(2022, 12, 31))
        assert all(e.meeting_start_date < e.announcement_date for e in events)

    def test_range_filter(self) -> None:
        # Jun 15, Jul 27, Sep 21, Nov 2, Dec 14
        events = load_fomc_announcements(date(2016, 6, 1), date(2016, 12, 31))
        assert len(events) == 5

    def test_sorted_ascending(self) -> None:
        events = load_fomc_announcements(date(2015, 1, 1), date(2022, 12, 31))
        dates = [e.announcement_date for e in events]
        assert dates == sorted(dates)

    def test_no_bounds_returns_all(self) -> None:
        events = load_fomc_announcements()
        assert len(events) >= 63

    def test_result_type(self) -> None:
        events = load_fomc_announcements(date(2015, 1, 1), date(2015, 12, 31))
        assert all(isinstance(e, FomcAnnouncement) for e in events)

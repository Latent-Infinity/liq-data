"""FOMC scheduled-announcement calendar (static, 2015-2022).

Source: federalreserve.gov meeting calendars, retrieved 2026-07-04:
- https://www.federalreserve.gov/monetarypolicy/fomchistorical2015.htm
  (and the analogous pages for 2016-2020)
- https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (2021-2022)

Scheduled meetings only. Statements are released at 14:00 ET (standard since
2013). Excluded as unscheduled: the 2019-10-04 conference call; the 2020-03-02
emergency call (statement 2020-03-03) and the 2020-03-15 emergency meeting,
which superseded the regularly scheduled March 2020 meeting (hence 2020 has
seven scheduled announcements); the 2020 notation votes (03-19, 03-23, 03-31,
08-27). All timestamps are converted to UTC via America/New_York (DST-aware);
consumers work in UTC and convert only at display boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import cached_property
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

ANNOUNCEMENT_TIME_ET = "14:00"

# (meeting_start, announcement_date) — announcement on the meeting's last day.
_SCHEDULED_MEETINGS: tuple[tuple[date, date], ...] = (
    (date(2015, 1, 27), date(2015, 1, 28)),
    (date(2015, 3, 17), date(2015, 3, 18)),
    (date(2015, 4, 28), date(2015, 4, 29)),
    (date(2015, 6, 16), date(2015, 6, 17)),
    (date(2015, 7, 28), date(2015, 7, 29)),
    (date(2015, 9, 16), date(2015, 9, 17)),
    (date(2015, 10, 27), date(2015, 10, 28)),
    (date(2015, 12, 15), date(2015, 12, 16)),
    (date(2016, 1, 26), date(2016, 1, 27)),
    (date(2016, 3, 15), date(2016, 3, 16)),
    (date(2016, 4, 26), date(2016, 4, 27)),
    (date(2016, 6, 14), date(2016, 6, 15)),
    (date(2016, 7, 26), date(2016, 7, 27)),
    (date(2016, 9, 20), date(2016, 9, 21)),
    (date(2016, 11, 1), date(2016, 11, 2)),
    (date(2016, 12, 13), date(2016, 12, 14)),
    (date(2017, 1, 31), date(2017, 2, 1)),
    (date(2017, 3, 14), date(2017, 3, 15)),
    (date(2017, 5, 2), date(2017, 5, 3)),
    (date(2017, 6, 13), date(2017, 6, 14)),
    (date(2017, 7, 25), date(2017, 7, 26)),
    (date(2017, 9, 19), date(2017, 9, 20)),
    (date(2017, 10, 31), date(2017, 11, 1)),
    (date(2017, 12, 12), date(2017, 12, 13)),
    (date(2018, 1, 30), date(2018, 1, 31)),
    (date(2018, 3, 20), date(2018, 3, 21)),
    (date(2018, 5, 1), date(2018, 5, 2)),
    (date(2018, 6, 12), date(2018, 6, 13)),
    (date(2018, 7, 31), date(2018, 8, 1)),
    (date(2018, 9, 25), date(2018, 9, 26)),
    (date(2018, 11, 7), date(2018, 11, 8)),
    (date(2018, 12, 18), date(2018, 12, 19)),
    (date(2019, 1, 29), date(2019, 1, 30)),
    (date(2019, 3, 19), date(2019, 3, 20)),
    (date(2019, 4, 30), date(2019, 5, 1)),
    (date(2019, 6, 18), date(2019, 6, 19)),
    (date(2019, 7, 30), date(2019, 7, 31)),
    (date(2019, 9, 17), date(2019, 9, 18)),
    (date(2019, 10, 29), date(2019, 10, 30)),
    (date(2019, 12, 10), date(2019, 12, 11)),
    (date(2020, 1, 28), date(2020, 1, 29)),
    (date(2020, 4, 28), date(2020, 4, 29)),
    (date(2020, 6, 9), date(2020, 6, 10)),
    (date(2020, 7, 28), date(2020, 7, 29)),
    (date(2020, 9, 15), date(2020, 9, 16)),
    (date(2020, 11, 4), date(2020, 11, 5)),
    (date(2020, 12, 15), date(2020, 12, 16)),
    (date(2021, 1, 26), date(2021, 1, 27)),
    (date(2021, 3, 16), date(2021, 3, 17)),
    (date(2021, 4, 27), date(2021, 4, 28)),
    (date(2021, 6, 15), date(2021, 6, 16)),
    (date(2021, 7, 27), date(2021, 7, 28)),
    (date(2021, 9, 21), date(2021, 9, 22)),
    (date(2021, 11, 2), date(2021, 11, 3)),
    (date(2021, 12, 14), date(2021, 12, 15)),
    (date(2022, 1, 25), date(2022, 1, 26)),
    (date(2022, 3, 15), date(2022, 3, 16)),
    (date(2022, 5, 3), date(2022, 5, 4)),
    (date(2022, 6, 14), date(2022, 6, 15)),
    (date(2022, 7, 26), date(2022, 7, 27)),
    (date(2022, 9, 20), date(2022, 9, 21)),
    (date(2022, 11, 1), date(2022, 11, 2)),
    (date(2022, 12, 13), date(2022, 12, 14)),
)


@dataclass(frozen=True)
class FomcAnnouncement:
    """One scheduled FOMC statement release."""

    meeting_start_date: date
    announcement_date: date
    announcement_time_et: str = ANNOUNCEMENT_TIME_ET

    @cached_property
    def announcement_utc(self) -> datetime:
        """Statement release instant in UTC (DST-aware)."""
        hour, minute = (int(part) for part in self.announcement_time_et.split(":"))
        eastern = datetime(
            self.announcement_date.year,
            self.announcement_date.month,
            self.announcement_date.day,
            hour,
            minute,
            tzinfo=_EASTERN,
        )
        return eastern.astimezone(_UTC)


def load_fomc_announcements(
    start: date | None = None,
    end: date | None = None,
) -> list[FomcAnnouncement]:
    """Scheduled FOMC announcements with announcement date in [start, end]."""
    events = [
        FomcAnnouncement(meeting_start_date=m_start, announcement_date=a_date)
        for m_start, a_date in _SCHEDULED_MEETINGS
        if (start is None or a_date >= start) and (end is None or a_date <= end)
    ]
    return sorted(events, key=lambda e: e.announcement_date)

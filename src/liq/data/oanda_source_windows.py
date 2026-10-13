"""Descriptive London-window gap membership, not scientific eligibility rules."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from liq.data.fx_session import LONDON_OPEN, london_open_utc
from liq.data.oanda_source_metadata import CandleDaySummary


@dataclass(frozen=True, slots=True)
class AnchorGaps:
    london_hour: int
    anchor_utc: datetime
    start_utc: datetime
    end_utc: datetime
    provider_absent: tuple[datetime, ...]
    raw_incomplete: tuple[datetime, ...]
    adapter_dropped: tuple[datetime, ...]


def describe_anchor_gaps(summary: CandleDaySummary) -> tuple[AnchorGaps, ...]:
    """Include both window edges for metadata inspection; make no fill/return choice."""
    opening = london_open_utc(summary.scope.day)
    windows: list[AnchorGaps] = []
    for hour in (14, 15, 16):
        anchor = opening + timedelta(hours=hour - LONDON_OPEN.hour)
        start, end = anchor - timedelta(minutes=20), anchor + timedelta(minutes=30)
        difference = summary.difference
        windows.append(
            AnchorGaps(
                hour,
                anchor,
                start,
                end,
                tuple(stamp for stamp in difference.provider_absent if start <= stamp <= end),
                tuple(stamp for stamp in difference.raw_incomplete if start <= stamp <= end),
                tuple(stamp for stamp in difference.adapter_dropped if start <= stamp <= end),
            )
        )
    return tuple(windows)

"""Calendar probes only; the timestamps below are not observed market gaps."""

from datetime import UTC, date, datetime, timedelta

import pytest

from liq.data.oanda_coverage import CoverageDifference
from liq.data.oanda_source_metadata import CandleDay, CandleDaySummary
from liq.data.oanda_source_windows import describe_anchor_gaps


@pytest.mark.parametrize("day,hour_utc", [(date(2024, 1, 3), 16), (date(2024, 7, 3), 15)])
def test_fix_and_control_windows_use_london_dst_and_retain_both_boundaries(
    day: date, hour_utc: int
) -> None:
    anchor = datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(hours=hour_utc)
    probes = tuple(anchor + timedelta(minutes=offset) for offset in (-21, -20, 0, 30, 31))
    summary = CandleDaySummary(
        CandleDay("EUR_USD", day), 0, 0, 0, (), CoverageDifference(probes, (), ())
    )

    windows = describe_anchor_gaps(summary)

    assert tuple(window.london_hour for window in windows) == (14, 15, 16)
    assert windows[-1].anchor_utc == anchor
    assert windows[-1].provider_absent == probes[1:4]
    assert windows[-1].start_utc == probes[1] and windows[-1].end_utc == probes[3]
    assert windows[0].anchor_utc == anchor - timedelta(hours=2)
    assert windows[-1].raw_incomplete == windows[-1].adapter_dropped == ()

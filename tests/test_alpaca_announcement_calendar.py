"""Nonfinancial refusal controls; no invented native calendar acceptance evidence."""

import json
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Literal, assert_never

import pytest

from liq.data.alpaca_calendar import CalendarRefused


def _metadata_control() -> bytes:
    """Hypothetical schedule control, never observed sessions or source evidence."""
    days = (
        "2022-04-25",
        "2022-04-26",
        "2022-04-27",
        "2022-07-26",
        "2022-07-27",
        "2022-10-25",
        "2022-10-26",
        "2023-02-14",
        "2023-02-15",
    )
    return json.dumps([{"date": day, "open": "09:30", "close": "16:00"} for day in days]).encode()


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b"x" * 131073, "byte_budget"),
        (b"not-json", "schema"),
        (b'{"date":null}', "schema"),
        (b"[]", "record_budget"),
        (b"[{}]", "schema"),
        (b"[null]", "schema"),
        (
            b'[ {"date":"invalid","date":"OTHER","open":"CONTROL","close":"CONTROL"} ]',
            "duplicate_json_key",
        ),
        (b'[ {"date":"invalid","open":"CONTROL","close":"CONTROL"} ]', "date"),
        (b'[ {"date":"20220425","open":"CONTROL","close":"CONTROL"} ]', "date_label"),
        (b'[ {"date":"2022-04-25","open":"CONTROL","close":"CONTROL"} ]', "invalid_clock"),
    ],
)
def test_invalid_calendar_bytes_refuse(raw: bytes, reason: str) -> None:
    from liq.data.alpaca_announcement_calendar import parse_announcement_calendar

    with pytest.raises(CalendarRefused, match=reason):
        parse_announcement_calendar(raw, datetime(2026, 10, 4, tzinfo=UTC))


@pytest.mark.parametrize(
    "clock",
    [
        datetime(2026, 10, 4),
        datetime(2026, 10, 4, tzinfo=timezone(timedelta(hours=1))),
        datetime(2023, 2, 14, tzinfo=UTC),
    ],
)
def test_capture_clock_is_checked_before_payload(clock: datetime) -> None:
    from liq.data.alpaca_announcement_calendar import parse_announcement_calendar

    with pytest.raises(CalendarRefused, match="capture_clock"):
        parse_announcement_calendar(b"not-json", clock)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_is_refused_even_in_unknown_fields(constant: str) -> None:
    from liq.data.alpaca_announcement_calendar import parse_announcement_calendar

    raw = _metadata_control().replace(
        b'"date":', b'"nonfinancial_control_extra":' + constant.encode() + b',"date":', 1
    )
    with pytest.raises(CalendarRefused, match="nonfinite_json"):
        parse_announcement_calendar(raw, datetime(2026, 10, 4, tzinfo=UTC))


def test_nonfinancial_control_preserves_labels_and_four_bindings() -> None:
    from liq.data.alpaca_announcement_calendar import parse_announcement_calendar

    result = parse_announcement_calendar(_metadata_control(), datetime(2026, 10, 4, tzinfo=UTC))
    assert [
        (pair.accession, pair.preceding.date, pair.following.date) for pair in result.event_pairs
    ] == [
        ("0001637459-22-000056", date(2022, 4, 26), date(2022, 4, 27)),
        ("0001637459-22-000156", date(2022, 7, 26), date(2022, 7, 27)),
        ("0001637459-22-000186", date(2022, 10, 25), date(2022, 10, 26)),
        ("0001637459-23-000005", date(2023, 2, 14), date(2023, 2, 15)),
    ]
    assert result.event_pairs[0].preceding.original_close == "16:00"
    assert result.event_pairs[0].preceding.close_utc == datetime(2022, 4, 26, 20, tzinfo=UTC)
    assert result.event_pairs[-1].following.original_open == "09:30"
    assert result.event_pairs[-1].following.open_utc == datetime(2023, 2, 15, 14, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "descending",
        "missing",
        "weekend",
        "scope",
        "interval",
        "last_open_early",
        "record_budget",
    ],
)
def test_metadata_defects_cannot_be_repaired(
    defect: Literal[
        "duplicate",
        "descending",
        "missing",
        "weekend",
        "scope",
        "interval",
        "last_open_early",
        "record_budget",
    ],
) -> None:
    from liq.data.alpaca_announcement_calendar import parse_announcement_calendar

    rows = json.loads(_metadata_control())
    match defect:
        case "duplicate":
            rows.insert(1, rows[1])
        case "descending":
            rows[1], rows[2] = rows[2], rows[1]
        case "missing":
            rows.pop(1)
        case "weekend":
            rows.insert(3, {"date": "2022-04-30", "open": "09:30", "close": "16:00"})
        case "scope":
            rows[0]["date"] = "2022-04-22"
        case "interval":
            rows[0]["close"] = "09:30"
        case "last_open_early":
            rows[-1]["open"] = "07:00"
        case "record_budget":
            rows *= 34
        case unreachable:
            assert_never(unreachable)
    reasons = {
        "duplicate": "date_scope",
        "descending": "date_scope",
        "missing": "event_pair_disagrees",
        "weekend": "date_scope",
        "scope": "date_scope",
        "interval": "session_interval",
        "last_open_early": "event_boundary_missing",
        "record_budget": "record_budget",
    }
    with pytest.raises(CalendarRefused, match=reasons[defect]):
        parse_announcement_calendar(json.dumps(rows).encode(), datetime(2026, 10, 4, tzinfo=UTC))


@pytest.mark.parametrize("side", ["close", "open"])
def test_metadata_clock_equality_is_excluded(monkeypatch: pytest.MonkeyPatch, side: str) -> None:
    """Hypothetical equality control; the frozen SEC clocks are not changed."""
    import liq.data.alpaca_announcement_calendar as module

    accession = "0001637459-22-000056"
    event = (
        datetime(2022, 4, 26, 20, tzinfo=UTC)
        if side == "close"
        else datetime(2022, 4, 27, 13, 30, tzinfo=UTC)
    )
    expected = (
        (date(2022, 4, 25), date(2022, 4, 27))
        if side == "close"
        else (date(2022, 4, 26), date(2022, 7, 26))
    )
    monkeypatch.setattr(module, "_EVENT_CLOCKS", ((accession, event),))
    monkeypatch.setattr(module, "ANNOUNCEMENT_DATE_PAIRS", {accession: expected})
    result = module.parse_announcement_calendar(
        _metadata_control(), datetime(2026, 10, 4, tzinfo=UTC)
    )
    pair = result.event_pairs[0]
    assert (pair.preceding.date, pair.following.date) == expected

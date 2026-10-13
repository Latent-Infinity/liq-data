"""Negative source-schema controls without fabricated market prices."""

import json
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from liq.data.alpaca_daily_fixture import DailyFixtureRefused, parse_daily_fixture
from liq.data.alpaca_source import AlpacaDailyFixtureRequest

_START, _END = date(2023, 1, 26), date(2023, 4, 24)
_DAYS = tuple(_START + timedelta(days=i) for i in range((_END - _START).days + 1))
# Source-free label metadata, not evidence of exchange-calendar admission.
LABELS = tuple(
    datetime.combine(day, datetime.min.time(), ZoneInfo("America/New_York")).astimezone(UTC)
    for day in _DAYS
    if day.weekday() < 5 and day not in {date(2023, 2, 20), date(2023, 4, 7)}
)
CAPTURED = datetime(2026, 10, 3, tzinfo=UTC)
REQUEST = AlpacaDailyFixtureRequest(symbol="KHC")


def test_parser_refuses_missing_bars() -> None:
    with pytest.raises(DailyFixtureRefused, match="schema"):
        parse_daily_fixture(b"{}", REQUEST, LABELS, CAPTURED)


def test_parser_refuses_duplicate_json_keys() -> None:
    with pytest.raises(DailyFixtureRefused, match="duplicate"):
        parse_daily_fixture(b'{"bars":{},"bars":{}}', REQUEST, LABELS, CAPTURED)


@pytest.mark.parametrize("raw", [b"not JSON", b"\xff"])
def test_parser_refuses_malformed_body(raw: bytes) -> None:
    with pytest.raises(DailyFixtureRefused, match="schema"):
        parse_daily_fixture(raw, REQUEST, LABELS, CAPTURED)


def test_parser_refuses_symbol_substitution() -> None:
    raw = json.dumps({"bars": {"SPY": []}, "next_page_token": None}).encode()
    with pytest.raises(DailyFixtureRefused, match="symbol"):
        parse_daily_fixture(raw, REQUEST, LABELS, CAPTURED)


def test_parser_refuses_incomplete_dates() -> None:
    raw = json.dumps({"bars": {"KHC": []}, "next_page_token": None}).encode()
    with pytest.raises(DailyFixtureRefused, match="61"):
        parse_daily_fixture(raw, REQUEST, LABELS, CAPTURED)


@pytest.mark.parametrize("token", ["next", ""])
def test_parser_refuses_any_continuation(token: str) -> None:
    raw = json.dumps({"bars": {"KHC": []}, "next_page_token": token}).encode()
    with pytest.raises(DailyFixtureRefused, match="continuation"):
        parse_daily_fixture(raw, REQUEST, LABELS, CAPTURED)


def test_parser_refuses_missing_close_tokens() -> None:
    # Timestamp metadata only: every price is absent.
    rows = [{"t": ts.isoformat().replace("+00:00", "Z"), "c": None} for ts in LABELS]
    raw = json.dumps({"bars": {"KHC": rows}, "next_page_token": None}).encode()
    with pytest.raises(DailyFixtureRefused, match="schema"):
        parse_daily_fixture(raw, REQUEST, LABELS, CAPTURED)


def test_parser_requires_exact_increasing_utc_label_contract() -> None:
    with pytest.raises(DailyFixtureRefused, match="label contract"):
        parse_daily_fixture(b"{}", REQUEST, tuple(reversed(LABELS)), CAPTURED)


def test_parser_refuses_shorter_label_contract() -> None:
    with pytest.raises(DailyFixtureRefused, match="label contract"):
        parse_daily_fixture(b"{}", REQUEST, LABELS[:-1], CAPTURED)


def test_parser_refuses_naive_capture_clock() -> None:
    with pytest.raises(DailyFixtureRefused, match="capture clock"):
        parse_daily_fixture(b"{}", REQUEST, LABELS, datetime(2026, 10, 3))


def test_parser_enforces_raw_byte_budget_before_decoding() -> None:
    with pytest.raises(DailyFixtureRefused, match="byte budget"):
        parse_daily_fixture(b" " * REQUEST.max_response_bytes, REQUEST, LABELS, CAPTURED)

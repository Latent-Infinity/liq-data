"""Approved source-capture time domains, without financial observations or transport."""

import json
from datetime import date, datetime

import pytest
from pydantic import ValidationError

from liq.data.oanda_source_capture_contract import SourcePriceWindow

PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")
RANGES = (
    (date(2020, 1, 3), "13:39:00Z", "16:29:59.999999Z"),
    (date(2024, 1, 3), "13:39:00Z", "16:29:59.999999Z"),
    (date(2024, 1, 31), "13:39:00Z", "16:29:59.999999Z"),
    (date(2024, 3, 15), "13:39:00Z", "16:29:59.999999Z"),
    (date(2024, 7, 3), "12:39:00Z", "15:29:59.999999Z"),
    (date(2024, 12, 30), "13:39:00Z", "16:29:59.999999Z"),
)


@pytest.mark.parametrize("symbol", PAIRS)
@pytest.mark.parametrize("case", RANGES)
def test_query_uses_exact_reviewed_utc_interval_when_case_is_approved(
    symbol: str, case: tuple[date, str, str]
) -> None:
    day, lower, upper = case
    window = SourcePriceWindow(symbol=symbol, day=day)

    query = window.query()

    assert query == {
        "from": f"{day}T{lower}",
        "to": f"{day}T{upper}",
        "price": "BA",
        "granularity": "M1",
        "units": "1",
        "smooth": "false",
        "includeFirst": "true",
    }


@pytest.mark.parametrize("symbol", ["BTC_USD", "EUR_USD/../orders", "", "eur_usd"])
def test_boundary_refuses_when_symbol_is_outside_reviewed_universe(symbol: str) -> None:
    with pytest.raises(ValidationError):
        SourcePriceWindow(symbol=symbol, day=date(2024, 1, 3))


@pytest.mark.parametrize(
    "day",
    [date(2019, 12, 31), date(2024, 3, 31), date(2024, 10, 27), date(2025, 1, 3)],
)
def test_boundary_refuses_when_date_is_not_a_reviewed_case(day: date) -> None:
    with pytest.raises(ValidationError):
        SourcePriceWindow(symbol="EUR_USD", day=day)


@pytest.mark.parametrize(
    "payload",
    [
        {"symbol": "EUR_USD", "day": "2024-01-03", "count": 171},
        {"symbol": "EUR_USD", "day": "2024-01-03", "price": "M"},
        {"symbol": "EUR_USD", "day": "2024-01-03", "from": "2024-01-03T13:40:00Z"},
        {"symbol": True, "day": "2024-01-03"},
        {"symbol": "EUR_USD", "day": False},
    ],
)
def test_json_boundary_refuses_when_scope_overrides_or_coercions_are_supplied(
    payload: dict[str, str | int | bool],
) -> None:
    with pytest.raises(ValidationError):
        SourcePriceWindow.model_validate_json(json.dumps(payload))


def test_json_boundary_parses_native_date_when_case_is_approved() -> None:
    payload = '{"symbol":"EUR_USD","day":"2024-01-03"}'

    window = SourcePriceWindow.model_validate_json(payload)

    assert window.symbol == "EUR_USD" and window.day == date(2024, 1, 3)


def test_boundary_refuses_when_datetime_is_supplied_instead_of_date() -> None:
    with pytest.raises(ValidationError):
        SourcePriceWindow.model_validate({"symbol": "EUR_USD", "day": datetime(2024, 1, 3, 0, 0)})


def test_window_is_immutable_when_caller_attempts_to_replace_day() -> None:
    window = SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))

    with pytest.raises(ValidationError):
        window.day = date(2025, 1, 3)

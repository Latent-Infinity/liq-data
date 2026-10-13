"""Development-source scope and clock contracts, without market observations or GETs."""

import json
from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_source_capture_contract import SourcePriceWindow

SCOPE = "london-fx-fix-development-source/v1"
PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")
EVALUATION_DAYS = (
    date(2024, 1, 31),
    date(2024, 3, 28),
    date(2024, 4, 2),
    date(2024, 7, 3),
    date(2024, 10, 25),
    date(2024, 10, 28),
)
CLOCK_CASES = (
    (date(2023, 1, 2), "13:39:00Z", "16:29:59.999999Z"),
    (date(2023, 3, 24), "13:39:00Z", "16:29:59.999999Z"),
    (date(2023, 3, 27), "12:39:00Z", "15:29:59.999999Z"),
    (date(2023, 10, 27), "12:39:00Z", "15:29:59.999999Z"),
    (date(2023, 10, 30), "13:39:00Z", "16:29:59.999999Z"),
    (date(2024, 1, 31), "13:39:00Z", "16:29:59.999999Z"),
    (date(2024, 3, 28), "13:39:00Z", "16:29:59.999999Z"),
    (date(2024, 4, 2), "12:39:00Z", "15:29:59.999999Z"),
    (date(2024, 7, 3), "12:39:00Z", "15:29:59.999999Z"),
    (date(2024, 10, 25), "12:39:00Z", "15:29:59.999999Z"),
    (date(2024, 10, 28), "13:39:00Z", "16:29:59.999999Z"),
)


@pytest.mark.parametrize("symbol", PAIRS)
@pytest.mark.parametrize("case", CLOCK_CASES)
def test_query_has_fixed_native_clock_and_parameters(
    symbol: str, case: tuple[date, str, str]
) -> None:
    day, lower, upper = case
    window = CohortSourceWindow(scope=SCOPE, symbol=symbol, day=day)

    assert window.query() == {
        "from": f"{day}T{lower}",
        "to": f"{day}T{upper}",
        "price": "BA",
        "granularity": "M1",
        "units": "1",
        "smooth": "false",
        "includeFirst": "true",
    }
    assert window.publication_status == "unverified"


def test_every_training_weekday_and_only_six_evaluation_dates_are_admitted() -> None:
    accepted = 0
    for year in (2023, 2024):
        day = date(year, 1, 1)
        while day.year == year:
            for symbol in PAIRS:
                if (year == 2023 and day.weekday() < 5) or day in EVALUATION_DAYS:
                    assert CohortSourceWindow(scope=SCOPE, symbol=symbol, day=day).day == day
                    accepted += 1
                else:
                    with pytest.raises(ValidationError):
                        CohortSourceWindow(scope=SCOPE, symbol=symbol, day=day)
            day += timedelta(days=1)

    assert accepted == 1862


@pytest.mark.parametrize(
    "day", [date(2019, 1, 2), date(2020, 1, 3), date(2025, 1, 2), date(2026, 1, 2)]
)
def test_other_years_are_not_admitted(day: date) -> None:
    with pytest.raises(ValidationError):
        CohortSourceWindow(scope=SCOPE, symbol="EUR_USD", day=day)


@pytest.mark.parametrize("symbol", ["BTC_USD", "EUR_USD/../orders", "", "eur_usd"])
def test_other_symbols_are_not_admitted(symbol: str) -> None:
    with pytest.raises(ValidationError):
        CohortSourceWindow(scope=SCOPE, symbol=symbol, day=date(2023, 1, 2))


@pytest.mark.parametrize(
    "changes",
    [
        {"scope": "other"},
        {"publication_status": "verified"},
        {"price": "M"},
        {"from": "2023-01-02T13:40:00Z"},
        {"units": "100"},
        {"count": 171},
        {"symbol": True},
        {"day": False},
    ],
)
def test_json_refuses_overrides_and_coercions(changes: dict[str, str | bool | int]) -> None:
    payload = {"scope": SCOPE, "symbol": "EUR_USD", "day": "2023-01-02"} | changes
    with pytest.raises(ValidationError):
        CohortSourceWindow.model_validate_json(json.dumps(payload))


def test_json_date_round_trip_preserves_unverified_publication_status() -> None:
    payload = json.dumps({"scope": SCOPE, "symbol": "EUR_USD", "day": "2023-01-02"})
    window = CohortSourceWindow.model_validate_json(payload)

    assert window.day == date(2023, 1, 2)
    assert CohortSourceWindow.model_validate_json(window.model_dump_json()) == window


@pytest.mark.parametrize("day", ["2023-01-02", datetime(2023, 1, 2)])
def test_python_input_requires_a_date_not_string_or_datetime(day: str | datetime) -> None:
    with pytest.raises(ValidationError):
        CohortSourceWindow.model_validate({"scope": SCOPE, "symbol": "EUR_USD", "day": day})


def test_window_is_immutable() -> None:
    window = CohortSourceWindow(scope=SCOPE, symbol="EUR_USD", day=date(2023, 1, 2))
    with pytest.raises(ValidationError):
        window.day = date(2024, 1, 31)


@pytest.mark.parametrize("construct", [False, True], ids=["copy", "construct"])
@pytest.mark.parametrize(
    "changes",
    [
        {"day": date(2025, 1, 2)},
        {"day": date(2026, 1, 2)},
        {"day": date(2024, 1, 3)},
        {"day": date(2023, 1, 1)},
        {"day": "2023-01-02"},
        {"symbol": "BTC_USD"},
        {"scope": "other"},
        {"publication_status": "verified"},
    ],
)
def test_query_revalidates_scope_after_unvalidated_creation(
    changes: dict[str, str | date],
    construct: bool,
) -> None:
    window = CohortSourceWindow(scope=SCOPE, symbol="EUR_USD", day=date(2023, 1, 2))
    forged = (
        CohortSourceWindow.model_construct(**(window.model_dump() | changes))
        if construct
        else window.model_copy(update=changes)
    )

    with pytest.raises(ValidationError):
        forged.query()


def test_legacy_and_cohort_payloads_are_not_interchangeable() -> None:
    legacy = SourcePriceWindow(symbol="AUD_USD", day=date(2024, 1, 31))
    cohort = CohortSourceWindow(scope=SCOPE, symbol="AUD_USD", day=date(2024, 1, 31))

    with pytest.raises(ValidationError):
        CohortSourceWindow.model_validate_json(legacy.model_dump_json())
    with pytest.raises(ValidationError):
        SourcePriceWindow.model_validate_json(cohort.model_dump_json())
    assert cohort.query() == legacy.query()


@pytest.mark.parametrize("day", [date(2023, 1, 2), date(2024, 3, 28), date(2024, 10, 28)])
def test_cohort_scope_does_not_expand_legacy_scope(day: date) -> None:
    CohortSourceWindow(scope=SCOPE, symbol="EUR_USD", day=day)
    with pytest.raises(ValidationError):
        SourcePriceWindow(symbol="EUR_USD", day=day)

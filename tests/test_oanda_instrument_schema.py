"""Strict native-unit metadata parsing; no financial observations are fabricated."""

import httpx
import pytest

from liq.data.oanda_instrument_metadata import summarize_instruments


def test_native_units_are_preserved_and_unrelated_fields_are_discarded() -> None:
    response = httpx.Response(
        200,
        json={
            "instruments": [
                {
                    "name": "EUR_USD",
                    "type": "CURRENCY",
                    "pipLocation": -4,
                    "displayPrecision": 5,
                    "unrelatedAccountField": "private",
                }
            ],
            "lastTransactionID": "private",
        },
    )

    observation = summarize_instruments(response)

    assert observation.category == "OBSERVED"
    assert observation.instruments[0].pip_location == -4
    assert observation.instruments[0].display_precision == 5
    assert "EUR_USD" not in observation.missing_symbols
    assert "private" not in observation.model_dump_json()


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipLocation", "-4"),
        ("pipLocation", True),
        ("displayPrecision", 5.0),
        ("displayPrecision", -1),
        ("pipLocation", -6),
        ("type", "CFD"),
        ("name", "BTC_USD"),
    ],
)
def test_invalid_native_units_and_symbols_are_rejected(
    field: str, value: str | int | float
) -> None:
    instrument: dict[str, str | int | float] = {
        "name": "EUR_USD",
        "type": "CURRENCY",
        "pipLocation": -4,
        "displayPrecision": 5,
    }
    instrument[field] = value

    observation = summarize_instruments(httpx.Response(200, json={"instruments": [instrument]}))

    assert observation.category == "SCHEMA_ERROR"
    assert observation.observed_at is None and observation.instruments == ()


@pytest.mark.parametrize("field", ["pipLocation", "displayPrecision"])
def test_absent_units_are_not_defaulted(field: str) -> None:
    instrument = {"name": "EUR_USD", "type": "CURRENCY", "pipLocation": -4, "displayPrecision": 5}
    del instrument[field]

    observation = summarize_instruments(httpx.Response(200, json={"instruments": [instrument]}))

    assert observation.category == "SCHEMA_ERROR"


def test_duplicate_symbol_is_a_source_anomaly() -> None:
    instrument = {"name": "EUR_USD", "type": "CURRENCY", "pipLocation": -4, "displayPrecision": 5}

    observation = summarize_instruments(
        httpx.Response(200, json={"instruments": [instrument, instrument]})
    )

    assert observation.category == "SCHEMA_ERROR" and observation.observed_at is None

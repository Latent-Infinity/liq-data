"""Nonnumeric schema probes verify native QA refusals, never market acceptance."""

from dataclasses import asdict
from datetime import UTC, date, datetime

import pytest
from test_oanda_source_envelope import EMPTY, NON_PRICE_CANDLE, SECRETS, WINDOW

from liq.data.oanda_source_capture_contract import PriceCaptureQuery, SourcePriceWindow
from liq.data.oanda_source_envelope import SourceEnvelopeRefused


@pytest.mark.parametrize("quarantined", [False, True])
@pytest.mark.parametrize("construct", [False, True])
@pytest.mark.parametrize(
    "day", [date(2023, 1, 2), date(2024, 3, 28), date(2025, 1, 2), date(2026, 1, 2)]
)
def test_legacy_quality_refuses_unvalidated_out_of_scope_windows(
    day: date, construct: bool, quarantined: bool
) -> None:
    from liq.data import oanda_source_quality as quality

    forged = (
        SourcePriceWindow.model_construct(symbol="EUR_USD", day=day)
        if construct
        else WINDOW.model_copy(update={"day": day})
    )

    with pytest.raises(SourceEnvelopeRefused) as error:
        if quarantined:
            quality.inspect_quarantined_source_quality(EMPTY, forged)
        else:
            quality.inspect_source_quality(EMPTY, forged, SECRETS)

    assert error.value.reason == "SCHEMA_REFUSED"
    assert error.value.__cause__ is None


def nonprice_body(*timestamps: str) -> bytes:
    candles = [
        NON_PRICE_CANDLE.replace(b"not-a-market-time", timestamp.encode())
        for timestamp in timestamps
    ]
    return b'{"instrument":"EUR_USD","granularity":"M1","candles":[' + b",".join(candles) + b"]}"


@pytest.mark.parametrize("quarantined", [False, True])
def test_legacy_projection_uses_reparsed_concrete_clock_not_subclass_override(
    quarantined: bool,
) -> None:
    from liq.data import oanda_source_quality as quality

    class WideClock(SourcePriceWindow):
        def query(self) -> PriceCaptureQuery:
            query = super().query()
            query["from"] = "2024-01-03T13:00:00Z"
            return query

    window = WideClock(symbol="EUR_USD", day=date(2024, 1, 3))
    body = nonprice_body("2024-01-03T13:38:00Z")

    with pytest.raises(quality.SourceQualityRefused) as error:
        if quarantined:
            quality.inspect_quarantined_source_quality(body, window)
        else:
            quality.inspect_source_quality(body, window, SECRETS)

    assert error.value.reason == "WINDOW_REFUSED"


@pytest.mark.parametrize("quarantined", [False, True])
def test_legacy_identity_snapshot_survives_caller_mutation_after_decode(
    quarantined: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from liq.data import oanda_source_envelope as envelopes
    from liq.data import oanda_source_quality as quality

    window = SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
    expected = SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
    original_parser = envelopes._parse_source_envelope
    mutated = False

    def decode_and_mutate(
        raw: bytes, symbol: str, credentials: tuple[str, str] | None
    ) -> envelopes._Envelope:
        nonlocal mutated
        envelope = original_parser(raw, symbol, credentials)
        object.__setattr__(window, "symbol", "GBP_USD")
        object.__setattr__(window, "day", date(2024, 7, 3))
        mutated = True
        return envelope

    with monkeypatch.context() as patch:
        patch.setattr(envelopes, "_parse_source_envelope", decode_and_mutate)
        if quarantined:
            result = quality.inspect_quarantined_source_quality(EMPTY, window)
        else:
            result = quality.inspect_source_quality(EMPTY, window, SECRETS)

    assert mutated
    assert result.window == expected
    assert result.candles == ()


def test_empty_native_projection_never_invents_a_candle() -> None:
    from liq.data import oanda_source_quality as quality

    result = quality.inspect_source_quality(EMPTY, WINDOW, SECRETS)

    assert result.window == WINDOW
    assert result.candles == ()


@pytest.mark.parametrize(
    "timestamp",
    ("2024-01-03T13:39:00.000000000Z", "2024-01-03T16:29:00+00:00"),
)
def test_exact_native_utc_forms_preserve_the_minute(timestamp: str) -> None:
    from liq.data import oanda_source_quality as quality

    result = quality.inspect_source_quality(nonprice_body(timestamp), WINDOW, SECRETS)

    assert result.candles[0].time == datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def test_summer_window_uses_the_contract_london_clock() -> None:
    from liq.data import oanda_source_quality as quality

    window = SourcePriceWindow(symbol="EUR_USD", day=date(2024, 7, 3))
    body = nonprice_body("2024-07-03T12:39:00Z", "2024-07-03T15:29:00Z")

    result = quality.inspect_source_quality(body, window, SECRETS)

    assert tuple(candle.time.hour for candle in result.candles) == (12, 15)


def test_a_missing_native_minute_is_not_filled() -> None:
    from liq.data import oanda_source_quality as quality

    body = nonprice_body("2024-01-03T13:39:00Z", "2024-01-03T13:41:00Z")

    result = quality.inspect_source_quality(body, WINDOW, SECRETS)

    assert tuple(candle.time.minute for candle in result.candles) == (39, 41)


@pytest.mark.parametrize("invalid", (b"NaN", b"sNaN", b"Infinity", b"-Infinity"))
def test_nonfinite_fields_remain_invalid_status_without_numeric_output(invalid: bytes) -> None:
    from liq.data import oanda_source_quality as quality

    body = nonprice_body("2024-01-03T13:39:00Z")
    for label in (b"open", b"high", b"low", b"close"):
        body = body.replace(b'"' + label + b'"', b'"' + invalid + b'"')

    result = quality.inspect_source_quality(body, WINDOW, SECRETS)

    expected = {"open": False, "high": False, "low": False, "close": False}
    assert asdict(result.candles[0].bid) == expected
    assert asdict(result.candles[0].ask) == expected


@pytest.mark.parametrize(
    "timestamp",
    (
        "not-a-native-time",
        "2024-01-03T13:39:00",
        "2024-01-03T13:39:00+01:00",
        "2024-01-03T13:39:01Z",
        "2024-01-03T13:39:00.000000001Z",
        "2024-02-30T13:39:00Z",
        "2024-01-03T25:39:00Z",
    ),
)
def test_native_timestamp_refusal_never_exposes_input(timestamp: str) -> None:
    from liq.data import oanda_source_quality as quality

    with pytest.raises(quality.SourceQualityRefused) as error:
        quality.inspect_source_quality(nonprice_body(timestamp), WINDOW, SECRETS)

    assert error.value.reason == "UTC_MINUTE_REFUSED"
    assert str(error.value) == "UTC_MINUTE_REFUSED"
    assert error.value.__cause__ is None


@pytest.mark.parametrize("timestamp", ("2024-01-03T13:38:00Z", "2024-01-03T16:30:00Z"))
def test_native_timestamp_outside_declared_window_is_refused(timestamp: str) -> None:
    from liq.data import oanda_source_quality as quality

    with pytest.raises(quality.SourceQualityRefused) as error:
        quality.inspect_source_quality(nonprice_body(timestamp), WINDOW, SECRETS)

    assert error.value.reason == "WINDOW_REFUSED"


@pytest.mark.parametrize("second", ("2024-01-03T13:40:00Z", "2024-01-03T13:39:00Z"))
def test_duplicate_or_reversed_native_stamps_are_never_sorted_into_validity(second: str) -> None:
    from liq.data import oanda_source_quality as quality

    with pytest.raises(quality.SourceQualityRefused) as error:
        quality.inspect_source_quality(
            nonprice_body("2024-01-03T13:40:00Z", second), WINDOW, SECRETS
        )

    assert error.value.reason == "ORDER_REFUSED"


def test_incomplete_candles_remain_in_original_native_order() -> None:
    from liq.data import oanda_source_quality as quality

    body = nonprice_body("2024-01-03T13:39:00Z", "2024-01-03T13:40:00Z")
    body = body.replace(b'"complete":false', b'"complete":true', 1)

    result = quality.inspect_source_quality(body, WINDOW, SECRETS)

    assert tuple(candle.complete for candle in result.candles) == (True, False)
    assert tuple(candle.time for candle in result.candles) == (
        datetime(2024, 1, 3, 13, 39, tzinfo=UTC),
        datetime(2024, 1, 3, 13, 40, tzinfo=UTC),
    )


def test_invalid_price_fields_produce_only_per_field_status() -> None:
    from liq.data import oanda_source_quality as quality

    result = quality.inspect_source_quality(nonprice_body("2024-01-03T13:39:00Z"), WINDOW, SECRETS)

    candle = result.candles[0]
    expected = {"open": False, "high": False, "low": False, "close": False}
    assert asdict(candle.bid) == expected
    assert asdict(candle.ask) == expected
    assert set(asdict(candle)) == {"time", "complete", "bid", "ask"}


@pytest.mark.parametrize("raw", (b"", b"not-json", EMPTY.replace(b"EUR_USD", b"GBP_USD")))
def test_native_quality_reuses_the_strict_envelope_refusal(raw: bytes) -> None:
    from liq.data import oanda_source_quality as quality

    with pytest.raises(SourceEnvelopeRefused) as error:
        quality.inspect_source_quality(raw, WINDOW, SECRETS)

    assert str(error.value) == "SCHEMA_REFUSED"


def test_native_quality_reuses_decoded_secret_refusal() -> None:
    from liq.data import oanda_source_quality as quality

    body = nonprice_body("dummy\\u002dtoken")

    with pytest.raises(SourceEnvelopeRefused) as error:
        quality.inspect_source_quality(body, WINDOW, SECRETS)

    assert str(error.value) == "SECRET_FOUND"

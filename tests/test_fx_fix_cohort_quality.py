"""Nonnumeric schema probes exercise cohort QA boundaries, not market evidence."""

from dataclasses import asdict
from datetime import UTC, date, datetime

import pytest
from test_oanda_source_envelope import EMPTY, SECRETS
from test_oanda_source_quality import nonprice_body

from liq.data.fx_fix_cohort_quality import (
    NativeCohortSourceQuality,
    inspect_cohort_source_quality,
    inspect_quarantined_cohort_source_quality,
)
from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_source_envelope import SourceEnvelopeRefused
from liq.data.oanda_source_quality import SourceQualityRefused

WINDOW = CohortSourceWindow(
    scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=date(2023, 1, 2)
)


@pytest.mark.parametrize("quarantined", [False, True])
def test_empty_quality_does_not_invent_candles_or_publication(quarantined: bool) -> None:
    if quarantined:
        result = inspect_quarantined_cohort_source_quality(EMPTY, WINDOW)
    else:
        result = inspect_cohort_source_quality(EMPTY, WINDOW, SECRETS)

    assert result == NativeCohortSourceQuality(window=WINDOW, candles=())
    assert result.window.publication_status == "unverified"


@pytest.mark.parametrize(
    "day,lower,upper",
    [
        (date(2023, 1, 2), "13:39:00", "16:29:00"),
        (date(2023, 3, 27), "12:39:00", "15:29:00"),
        (date(2024, 10, 25), "12:39:00", "15:29:00"),
        (date(2024, 10, 28), "13:39:00", "16:29:00"),
    ],
)
def test_native_projection_uses_cohort_dst_bounds_without_prices(
    day: date, lower: str, upper: str
) -> None:
    window = WINDOW.model_copy(update={"day": day})
    body = nonprice_body(f"{day}T{lower}Z", f"{day}T{upper}Z")

    result = inspect_cohort_source_quality(body, window, SECRETS)

    assert tuple(candle.time for candle in result.candles) == (
        datetime.fromisoformat(f"{day}T{lower}+00:00"),
        datetime.fromisoformat(f"{day}T{upper}+00:00"),
    )
    assert result.window == window


def test_invalid_fields_produce_only_boolean_status_and_original_completeness() -> None:
    body = nonprice_body("2023-01-02T13:39:00Z", "2023-01-02T13:41:00Z")
    body = body.replace(b'"complete":false', b'"complete":true', 1)

    result = inspect_cohort_source_quality(body, WINDOW, SECRETS)

    assert tuple(candle.complete for candle in result.candles) == (True, False)
    assert tuple(candle.time.minute for candle in result.candles) == (39, 41)
    assert result.candles[0].time.tzinfo is UTC
    for candle in result.candles:
        assert asdict(candle.bid) == {"open": False, "high": False, "low": False, "close": False}
        assert asdict(candle.ask) == {"open": False, "high": False, "low": False, "close": False}
        assert set(asdict(candle)) == {"time", "complete", "bid", "ask"}


@pytest.mark.parametrize("quarantined", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"day": date(2025, 1, 2)},
        {"day": date(2023, 1, 1)},
        {"day": "2023-01-02"},
        {"symbol": "GBP_JPY"},
        {"scope": "legacy"},
        {"publication_status": "verified"},
    ],
)
def test_quality_reparses_scope_before_body_inspection(
    changes: dict[str, str | date], quarantined: bool
) -> None:
    forged = WINDOW.model_copy(update=changes)

    with pytest.raises(SourceQualityRefused) as error:
        if quarantined:
            inspect_quarantined_cohort_source_quality(b"not-json", forged)
        else:
            inspect_cohort_source_quality(b"not-json", forged, SECRETS)

    assert error.value.reason == "WINDOW_REFUSED"
    assert str(error.value) == "WINDOW_REFUSED"
    assert error.value.__cause__ is None


def test_missing_scope_tag_is_not_a_legacy_payload_shortcut() -> None:
    forged = CohortSourceWindow.model_construct(symbol="EUR_USD", day=date(2024, 1, 31))

    with pytest.raises(SourceQualityRefused, match="WINDOW_REFUSED"):
        inspect_cohort_source_quality(EMPTY, forged, SECRETS)


@pytest.mark.parametrize(
    "raw",
    [b"", b"not-json", b"\xff", EMPTY.replace(b"EUR_USD", b"GBP_USD"), EMPTY.replace(b"M1", b"M5")],
)
def test_cohort_quality_keeps_the_strict_source_schema(raw: bytes) -> None:
    with pytest.raises(SourceEnvelopeRefused) as error:
        inspect_cohort_source_quality(raw, WINDOW, SECRETS)

    assert error.value.reason == "SCHEMA_REFUSED"
    assert error.value.__cause__ is None


@pytest.mark.parametrize("escaped", [False, True])
def test_secret_checks_remain_before_quality_projection(escaped: bool) -> None:
    secret = "dummy\\u002dtoken" if escaped else "dummy-token"
    body = nonprice_body(secret)

    with pytest.raises(SourceEnvelopeRefused) as error:
        inspect_cohort_source_quality(body, WINDOW, SECRETS)

    assert error.value.reason == "SECRET_FOUND"
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "timestamp,reason",
    [
        ("2023-01-02T13:39:01Z", "UTC_MINUTE_REFUSED"),
        ("2023-01-02T13:39:00+01:00", "UTC_MINUTE_REFUSED"),
        ("2023-02-30T13:39:00Z", "UTC_MINUTE_REFUSED"),
        ("2023-01-02T13:38:00Z", "WINDOW_REFUSED"),
        ("2023-01-02T16:30:00Z", "WINDOW_REFUSED"),
    ],
)
def test_invalid_native_clock_is_refused_without_parser_details(
    timestamp: str, reason: str
) -> None:
    with pytest.raises(SourceQualityRefused) as error:
        inspect_cohort_source_quality(nonprice_body(timestamp), WINDOW, SECRETS)

    assert error.value.reason == reason
    assert str(error.value) == reason
    assert error.value.__cause__ is None


@pytest.mark.parametrize("second", ["2023-01-02T13:40:00Z", "2023-01-02T13:39:00Z"])
def test_duplicates_and_descents_are_not_sorted_into_validity(second: str) -> None:
    body = nonprice_body("2023-01-02T13:40:00Z", second)

    with pytest.raises(SourceQualityRefused) as error:
        inspect_cohort_source_quality(body, WINDOW, SECRETS)

    assert error.value.reason == "ORDER_REFUSED"

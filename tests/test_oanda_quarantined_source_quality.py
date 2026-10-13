"""Post-quarantine schema probes, not fixture admission or price-quality evidence."""

from dataclasses import asdict

import pytest
from test_oanda_source_envelope import EMPTY, NON_PRICE_CANDLE, WINDOW, envelope_for

from liq.data.oanda_source_envelope import SourceEnvelopeRefused


def test_quarantined_empty_projection_does_not_admit_a_source() -> None:
    from liq.data import oanda_source_quality as quality

    result = quality.inspect_quarantined_source_quality(EMPTY, WINDOW)

    assert result.window == WINDOW
    assert result.candles == ()


def test_quarantined_projection_retains_incomplete_invalid_fields_without_values() -> None:
    from liq.data import oanda_source_quality as quality

    body = envelope_for(NON_PRICE_CANDLE.replace(b"not-a-market-time", b"2024-01-03T13:39:00Z"))

    result = quality.inspect_quarantined_source_quality(body, WINDOW)

    assert result.candles[0].complete is False
    expected = {"open": False, "high": False, "low": False, "close": False}
    assert asdict(result.candles[0].bid) == expected
    assert asdict(result.candles[0].ask) == expected


@pytest.mark.parametrize(
    "raw",
    (
        b"",
        b"not-json",
        EMPTY.replace(b"EUR_USD", b"GBP_USD"),
        EMPTY.replace(b'"candles": []', b'"candles": [], "candles": []'),
        EMPTY.replace(b'"candles": []', b'"candles": NaN'),
        b"[" * 2000,
    ),
)
def test_quarantined_projection_preserves_strict_envelope_refusals(raw: bytes) -> None:
    from liq.data import oanda_source_quality as quality

    with pytest.raises(SourceEnvelopeRefused) as error:
        quality.inspect_quarantined_source_quality(raw, WINDOW)

    assert error.value.reason == "SCHEMA_REFUSED"
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    ("timestamp", "reason"),
    (
        ("not-native-time", "UTC_MINUTE_REFUSED"),
        ("2024-01-03T13:39:00.000000001Z", "UTC_MINUTE_REFUSED"),
        ("2024-02-30T13:39:00Z", "UTC_MINUTE_REFUSED"),
        ("2024-01-03T13:38:00Z", "WINDOW_REFUSED"),
    ),
)
def test_quarantined_projection_preserves_native_refusals(timestamp: str, reason: str) -> None:
    from liq.data import oanda_source_quality as quality

    body = envelope_for(NON_PRICE_CANDLE.replace(b"not-a-market-time", timestamp.encode()))

    with pytest.raises(quality.SourceQualityRefused) as error:
        quality.inspect_quarantined_source_quality(body, WINDOW)

    assert error.value.reason == reason
    assert str(error.value) == reason
    assert error.value.__cause__ is None


def test_quarantined_projection_does_not_claim_a_second_secret_scan() -> None:
    from liq.data import oanda_source_quality as quality

    probe = NON_PRICE_CANDLE.replace(b"not-a-market-time", b"2024-01-03T13:39:00Z")
    body = envelope_for(probe.replace(b'"open"', b'"dummy-token"'))

    result = quality.inspect_quarantined_source_quality(body, WINDOW)

    assert result.candles[0].bid.open is False
    assert "dummy-token" not in repr(result)

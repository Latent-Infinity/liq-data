"""Empty/non-price envelopes test quarantine security, not market acceptance."""

from datetime import date

import pytest

from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.oanda_source_envelope import (
    CaptureSecrets,
    SourceEnvelopeRefused,
    check_source_envelope,
)

WINDOW = SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
SECRETS = CaptureSecrets(account_id="dummy-account", api_token="dummy-token")
EMPTY = b'{ "instrument": "EUR_USD", "granularity": "M1", "candles": [] }\n'
# Deliberately non-numeric schema probes, never prices or market fixtures.
NON_PRICE_CANDLE = (
    b'{"time":"not-a-market-time","bid":{"o":"open","h":"high","l":"low","c":"close"},'
    b'"ask":{"o":"open","h":"high","l":"low","c":"close"},"volume":0,"complete":false}'
)


def envelope_for(candle: bytes) -> bytes:
    return b'{"instrument":"EUR_USD","granularity":"M1","candles":[' + candle + b"]}"


def test_empty_envelope_only_satisfies_structural_quarantine_check() -> None:
    # Given no market values and an exact approved identity.
    # When schema and secret checks complete.
    result = check_source_envelope(EMPTY, WINDOW, SECRETS)
    # Then no quality verdict, price object or acceptance permission is returned.
    assert result is None


def test_structural_check_does_not_masquerade_as_price_quality() -> None:
    assert check_source_envelope(envelope_for(NON_PRICE_CANDLE), WINDOW, SECRETS) is None


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not-json",
        b"\xff",
        b"[]",
        b'{"instrument":"EUR_USD","granularity":"M1","candles":[],"extra":true}',
        EMPTY.replace(b"EUR_USD", b"GBP_USD"),
        EMPTY.replace(b"M1", b"M5"),
        EMPTY.replace(b'"candles": []', b'"candles": null'),
        EMPTY.replace(b'"candles": []', b'"candles": {}'),
        EMPTY.replace(b'"instrument": "EUR_USD", ', b""),
        EMPTY.replace(b'"candles": []', b'"candles": [],"candles":[]'),
        EMPTY.replace(b'"candles": []', b'"candles": NaN'),
        EMPTY.replace(b'"candles": []', b'"candles": Infinity'),
        EMPTY.replace(b'"candles": []', b'"candles": -Infinity'),
        envelope_for(NON_PRICE_CANDLE.replace(b'"volume":0', b'"volume":true')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"volume":0', b'"volume":-1')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"volume":0', b'"volume":"0"')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"volume":0', b'"volume":0.0')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"complete":false', b'"complete":0')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"complete":false', b'"complete":"false"')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"time":"not-a-market-time"', b'"time":0')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"bid":{', b'"bid":{"extra":"unlisted",')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"ask":{', b'"ask":{"c":"duplicate",')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"o":"open"', b'"o":null')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"o":"open",', b"")),
        envelope_for(NON_PRICE_CANDLE.replace(b'"l":"low"', b'"low":"low"')),
        envelope_for(NON_PRICE_CANDLE.replace(b'"complete":false', b'"complete":false,"mid":{}')),
        envelope_for(b"null"),
        b"[" * 2000 + b"]" * 2000,
        b"x" * 2097153,
    ],
)
def test_schema_refusal_is_typed_without_raw_payload(raw: bytes) -> None:
    with pytest.raises(SourceEnvelopeRefused) as error:
        check_source_envelope(raw, WINDOW, SECRETS)
    assert error.value.reason == "SCHEMA_REFUSED"
    assert str(error.value) == "SCHEMA_REFUSED"
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "raw",
    [
        EMPTY.replace(b"EUR_USD", b"dummy-account"),
        envelope_for(NON_PRICE_CANDLE.replace(b"not-a-market-time", b"dummy-token")),
        envelope_for(NON_PRICE_CANDLE.replace(b"not-a-market-time", b"dummy\\u002dtoken")),
        envelope_for(NON_PRICE_CANDLE.replace(b"not-a-market-time", b"dummy\\u002daccount")),
        EMPTY.replace(b'"candles": []', b'"candles": [],"dummy\\u002dtoken":[]'),
        EMPTY.replace(b'"candles": []', b'"candles": [],"extra":[["dummy\\u002dtoken"]]'),
        b"malformed dummy-token body",
    ],
)
def test_secret_scan_covers_raw_and_json_decoded_strings(raw: bytes) -> None:
    with pytest.raises(SourceEnvelopeRefused) as error:
        check_source_envelope(raw, WINDOW, SECRETS)
    assert error.value.reason == "SECRET_FOUND"
    assert "dummy" not in str(error.value)


@pytest.mark.parametrize("which", ["account_id", "api_token"])
def test_missing_secret_blocks_security_check(which: str) -> None:
    account = "" if which == "account_id" else "dummy-account"
    token = "" if which == "api_token" else "dummy-token"
    with pytest.raises(SourceEnvelopeRefused) as error:
        CaptureSecrets(account_id=account, api_token=token)
    assert error.value.reason == "INVALID_SECRETS"


def test_secret_record_repr_never_contains_credentials() -> None:
    assert "dummy" not in repr(SECRETS)

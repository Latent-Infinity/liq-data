"""Allowlisted source-price quarantine checks, separate from market quality."""

import json
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.providers.oanda_source_body import MAX_SOURCE_BODY_BYTES

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
EnvelopeFailureReason = Literal["SCHEMA_REFUSED", "SECRET_FOUND", "INVALID_SECRETS"]


class SourceEnvelopeRefused(ValueError):
    """A safe category only; never attach the raw payload or parser error."""

    def __init__(self, reason: EnvelopeFailureReason) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CaptureSecrets:
    """Ephemeral exact account/token strings, never persisted in receipts."""

    account_id: str = field(repr=False)
    api_token: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.account_id, str)
            or not isinstance(self.api_token, str)
            or not self.account_id
            or not self.api_token
        ):
            raise SourceEnvelopeRefused("INVALID_SECRETS")


class _EnvelopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _Prices(_EnvelopeModel):
    o: str
    h: str
    low: str = Field(alias="l")
    c: str


class _Candle(_EnvelopeModel):
    time: str
    bid: _Prices
    ask: _Prices
    volume: int = Field(ge=0)
    complete: bool


class _Envelope(_EnvelopeModel):
    instrument: str
    granularity: Literal["M1"]
    candles: list[_Candle]


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise SourceEnvelopeRefused("SCHEMA_REFUSED")
        result[key] = value
    return result


def _refuse_constant(_value: str) -> None:
    raise SourceEnvelopeRefused("SCHEMA_REFUSED")


def _contains_secret(value: JsonValue, secrets: tuple[str, str]) -> bool:
    if isinstance(value, str):
        return any(secret in value for secret in secrets)
    if isinstance(value, list):
        return any(_contains_secret(item, secrets) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_secret(key, secrets) or _contains_secret(item, secrets)
            for key, item in value.items()
        )
    return False


def check_source_envelope(raw: bytes, window: SourcePriceWindow, secrets: CaptureSecrets) -> None:
    """Check complete identity bytes before private quarantine retention.

    This makes no API call, writes nothing and admits no market observation.
    Ordered native timestamps, finite OHLC, crosses and prior missingness
    require separate source-quality reconciliation of the retained bytes.
    """
    _decode_source_envelope(raw, window, secrets)


def _decode_source_envelope(
    raw: bytes, window: SourcePriceWindow, secrets: CaptureSecrets
) -> _Envelope:
    """Share the bounded, secret-safe structural boundary with native quality."""
    validated = _reparse_legacy_window(window)
    return _parse_source_envelope(raw, validated.symbol, (secrets.account_id, secrets.api_token))


def _decode_quarantined_source_envelope(raw: bytes, window: SourcePriceWindow) -> _Envelope:
    """Reparse verified original quarantine only; this is not a secret scan."""
    validated = _reparse_legacy_window(window)
    return _parse_source_envelope(raw, validated.symbol, None)


def _reparse_legacy_window(window: SourcePriceWindow) -> SourcePriceWindow:
    try:
        return SourcePriceWindow.model_validate(window.model_dump(warnings=False))
    except ValidationError:
        raise SourceEnvelopeRefused("SCHEMA_REFUSED") from None


def _parse_source_envelope(
    raw: bytes, symbol: str, credentials: tuple[str, str] | None
) -> _Envelope:
    if len(raw) > MAX_SOURCE_BODY_BYTES:
        raise SourceEnvelopeRefused("SCHEMA_REFUSED")
    if credentials is not None and any(secret.encode("utf-8") in raw for secret in credentials):
        raise SourceEnvelopeRefused("SECRET_FOUND")
    try:
        payload: JsonValue = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_refuse_constant
        )
        if credentials is not None and _contains_secret(payload, credentials):
            raise SourceEnvelopeRefused("SECRET_FOUND")
        envelope = _Envelope.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, RecursionError):
        raise SourceEnvelopeRefused("SCHEMA_REFUSED") from None
    if envelope.instrument != symbol:
        raise SourceEnvelopeRefused("SCHEMA_REFUSED")
    return envelope

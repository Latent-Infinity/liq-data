"""Outcome-free native candle metadata; never source admission or market rows."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Final, Literal

from liq.data import oanda_source_envelope as envelopes
from liq.data.oanda_source_capture_contract import PriceCaptureQuery, SourcePriceWindow
from liq.data.oanda_source_envelope import (
    CaptureSecrets,
    _Envelope,
    _reparse_legacy_window,
)

QualityFailureReason = Literal["UTC_MINUTE_REFUSED", "WINDOW_REFUSED", "ORDER_REFUSED"]
_UTC_MINUTE: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)")


class SourceQualityRefused(ValueError):
    """A category-only refusal, without timestamps, prices or parser details."""

    def __init__(self, reason: QualityFailureReason) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class PriceFieldStatus:
    """Finite-positive status for each field, without retaining its value."""

    open: bool
    high: bool
    low: bool
    close: bool


@dataclass(frozen=True, slots=True)
class NativeCandleQuality:
    """Original completeness and native time, including incomplete candles."""

    time: datetime
    complete: bool
    bid: PriceFieldStatus
    ask: PriceFieldStatus


@dataclass(frozen=True, slots=True)
class NativeSourceQuality:
    """A projection only; callers separately reconcile evidence and missingness."""

    window: SourcePriceWindow
    candles: tuple[NativeCandleQuality, ...]


def _finite_positive(value: str) -> bool:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return False
    return number.is_finite() and number > 0


def inspect_source_quality(
    raw: bytes, window: SourcePriceWindow, secrets: CaptureSecrets
) -> NativeSourceQuality:
    """Inspect every source candle in order without patching or exposing prices."""
    validated = _reparse_legacy_window(window)
    query = validated.query()
    envelope = envelopes._parse_source_envelope(
        raw, validated.symbol, (secrets.account_id, secrets.api_token)
    )
    return NativeSourceQuality(window=validated, candles=_project_source_candles(envelope, query))


def inspect_quarantined_source_quality(
    raw: bytes, window: SourcePriceWindow
) -> NativeSourceQuality:
    """Recompute quality only after independent original-quarantine verification.

    The caller must restore the original private case, body SHA/state and bound
    secret-safe quarantine witnesses first. This performs no independent secret
    scan and must never replace the credential-bearing pre-retention boundary.
    """
    validated = _reparse_legacy_window(window)
    query = validated.query()
    envelope = envelopes._parse_source_envelope(raw, validated.symbol, None)
    return NativeSourceQuality(window=validated, candles=_project_source_candles(envelope, query))


def _project_source_candles(
    envelope: _Envelope, query: PriceCaptureQuery
) -> tuple[NativeCandleQuality, ...]:
    """Project only ordered native clock, completeness and unary field statuses."""
    lower = datetime.fromisoformat(query["from"].replace("Z", "+00:00"))
    upper = datetime.fromisoformat(query["to"].replace("Z", "+00:00"))
    previous: datetime | None = None
    candles: list[NativeCandleQuality] = []
    for candle in envelope.candles:
        if _UTC_MINUTE.fullmatch(candle.time) is None:
            raise SourceQualityRefused("UTC_MINUTE_REFUSED")
        try:
            timestamp = datetime.fromisoformat(candle.time.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            raise SourceQualityRefused("UTC_MINUTE_REFUSED") from None
        if not lower <= timestamp <= upper:
            raise SourceQualityRefused("WINDOW_REFUSED")
        if previous is not None and timestamp <= previous:
            raise SourceQualityRefused("ORDER_REFUSED")
        candles.append(
            NativeCandleQuality(
                time=timestamp,
                complete=candle.complete,
                bid=PriceFieldStatus(
                    open=_finite_positive(candle.bid.o),
                    high=_finite_positive(candle.bid.h),
                    low=_finite_positive(candle.bid.low),
                    close=_finite_positive(candle.bid.c),
                ),
                ask=PriceFieldStatus(
                    open=_finite_positive(candle.ask.o),
                    high=_finite_positive(candle.ask.h),
                    low=_finite_positive(candle.ask.low),
                    close=_finite_positive(candle.ask.c),
                ),
            )
        )
        previous = timestamp
    return tuple(candles)

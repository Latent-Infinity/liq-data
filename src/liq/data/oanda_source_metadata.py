"""Minimized bid/ask candle coverage summaries, without source-access authorization."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, assert_never

import httpx
from pydantic import StrictStr
from pydantic import ValidationError as PayloadValidationError

from liq.data.exceptions import ValidationError
from liq.data.providers.oanda_candles import _bid_ask_rows, _BidAskEnvelope

if TYPE_CHECKING:
    from liq.data.oanda_coverage import CoverageDifference


@dataclass(frozen=True, slots=True)
class CandleDay:
    """Expected response identity, not permission to fetch this pair or date."""

    symbol: str
    day: date


@dataclass(frozen=True, slots=True)
class CandleDaySummary:
    """Raw-versus-accepted minute coverage with no financial values."""

    scope: CandleDay
    raw_count: int
    complete_count: int
    accepted_count: int
    accepted_timestamps: tuple[datetime, ...]
    difference: "CoverageDifference"


@dataclass(frozen=True, slots=True)
class CandleDayObservation:
    """Transport/parse disposition; a missing summary is never an empty market day."""

    category: Literal["OBSERVED", "HTTP_ERROR", "TRANSPORT_ERROR", "SCHEMA_ERROR"]
    http_status: int | None
    summary: CandleDaySummary | None = None


class _DayEnvelope(_BidAskEnvelope):
    instrument: StrictStr
    granularity: Literal["M1"]


def summarize_candle_day(response: httpx.Response, scope: CandleDay) -> CandleDaySummary:
    """Validate complete BA candles and preserve every missing-minute explanation."""
    # The existing comparison module imports DataService for its older capture
    # API. Defer importing it until execution to keep provider imports acyclic.
    from liq.data.oanda_coverage import (
        RequestBoundaryError,
        compare_timestamps,
        extract_candle_metadata,
    )

    if response.status_code != 200:
        raise ValidationError("OANDA source-summary request was unsuccessful")
    start = datetime.combine(scope.day, time(), tzinfo=UTC)
    end = start + timedelta(days=1) - timedelta(microseconds=1)
    try:
        payload = _DayEnvelope.model_validate_json(response.content)
        if payload.instrument != scope.symbol:
            raise ValidationError("OANDA source-summary instrument mismatch")
        raw = extract_candle_metadata(response)
        if tuple(item.time for item in raw) != tuple(sorted(item.time for item in raw)):
            raise ValidationError("OANDA source-summary timestamps are unordered")
        rows = _bid_ask_rows(payload, start, end)
        accepted: list[datetime] = []
        for row in rows:
            match row["timestamp"]:
                case datetime() as timestamp:
                    accepted.append(timestamp)
                case Decimal():
                    raise ValidationError("OANDA decoder returned a non-timestamp identity")
                case unreachable:
                    assert_never(unreachable)
        difference = compare_timestamps(raw, tuple(accepted), start, end)
    except (PayloadValidationError, RequestBoundaryError):
        raise ValidationError("Invalid OANDA source-summary payload") from None
    return CandleDaySummary(
        scope=scope,
        raw_count=len(raw),
        complete_count=sum(item.complete for item in raw),
        accepted_count=len(accepted),
        accepted_timestamps=tuple(accepted),
        difference=difference,
    )

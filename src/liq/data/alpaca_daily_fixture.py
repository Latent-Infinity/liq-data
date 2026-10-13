"""Decode a retained daily arithmetic fixture without admitting historical use."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from liq.data.alpaca_source import AlpacaDailyFixtureRequest

JsonValue: TypeAlias = str | Decimal | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class DailyFixtureRefused(ValueError):
    """A bounded daily fixture has unresolved identity, labels or close fields."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _Close(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)
    t: StrictStr
    c: Decimal = Field(gt=0, allow_inf_nan=False)


class _Page(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", frozen=True)
    bars: dict[str, list[_Close]]
    next_page_token: StrictStr | None = None


@dataclass(frozen=True, slots=True)
class DailyFixture:
    """Original daily labels and current-vintage closes, known at capture time.

    Labels denote calendar-day starts, not closing auctions or availability.
    Source/calendar reconciliation and the exact capture witness belong to the
    admitted wrapper. This record alone confers no historical research permission.
    """

    request: AlpacaDailyFixtureRequest
    original_labels: tuple[str, ...]
    expected_labels: tuple[datetime, ...]
    closes: tuple[Decimal, ...]
    source_sha256: str
    available_at: datetime


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise DailyFixtureRefused("duplicate daily fixture JSON key")
        result[key] = value
    return result


def parse_daily_fixture(
    raw: bytes,
    request: AlpacaDailyFixtureRequest,
    expected_labels: tuple[datetime, ...],
    captured_at: datetime,
) -> DailyFixture:
    """Require exact61 ordered source labels, one symbol, no continuation and real c.

    Completed raw bytes must be retained before this call. The caller freezes the
    entire expected label list before prices and supplies completed-capture UTC
    time. Missing values or dates are refused rather than repaired or substituted.
    """
    if len(raw) >= request.max_response_bytes:
        raise DailyFixtureRefused("daily fixture byte budget reached")
    if (
        len(expected_labels) != 61
        or any(label.utcoffset() != timedelta(0) for label in expected_labels)
        or any(left >= right for left, right in pairwise(expected_labels))
        or expected_labels[0] != request.start
        or expected_labels[-1] != request.end
    ):
        raise DailyFixtureRefused("daily fixture label contract is incomplete or incoherent")
    if captured_at.utcoffset() != timedelta(0) or captured_at < expected_labels[-1]:
        raise DailyFixtureRefused("daily fixture capture clock requires honest completed UTC time")
    try:
        payload = json.loads(
            raw, parse_float=Decimal, parse_int=Decimal, object_pairs_hook=_unique_object
        )
        page = _Page.model_validate(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
        raise DailyFixtureRefused("daily fixture schema is invalid") from None
    if page.next_page_token is not None:
        raise DailyFixtureRefused("daily fixture continuation is forbidden")
    if set(page.bars) != {request.symbol}:
        raise DailyFixtureRefused("daily fixture symbol does not match request")
    rows = page.bars[request.symbol]
    if len(rows) != 61:
        raise DailyFixtureRefused("daily fixture requires exactly61 close records")
    labels = tuple(label.isoformat().replace("+00:00", "Z") for label in expected_labels)
    original = tuple(row.t for row in rows)
    if original != labels:
        raise DailyFixtureRefused("daily fixture dates are missing, extra, duplicated or unordered")
    return DailyFixture(
        request,
        original,
        expected_labels,
        tuple(row.c for row in rows),
        hashlib.sha256(raw).hexdigest(),
        captured_at,
    )

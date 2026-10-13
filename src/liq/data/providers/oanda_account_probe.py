"""Metadata projection for a bounded OANDA account-candle compatibility request."""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Literal
from urllib.parse import quote

import httpx
from pydantic import ValidationError as PayloadValidationError

from liq.data.exceptions import ValidationError
from liq.data.providers.oanda_candles import _bid_ask_rows, _BidAskEnvelope

PROBE_START: Final = datetime(2024, 1, 3, tzinfo=UTC)
PROBE_ARM: Final = "idea_05b_london_fix_source_verification"


@dataclass(frozen=True, slots=True)
class AccountCandleObservation:
    """Safe result with no rates, account identity, request object, or error body."""

    category: Literal[
        "OBSERVED", "EMPTY", "INCOMPLETE", "HTTP_ERROR", "TRANSPORT_ERROR", "SCHEMA_ERROR"
    ]
    http_status: int | None
    raw_count: int | None = None
    complete_count: int | None = None
    timestamps: tuple[datetime, ...] = ()


class _AccountEnvelope(_BidAskEnvelope):
    instrument: Literal["EUR_USD"]
    granularity: Literal["M1"]


class _AccountPathRedactor(logging.Filter):
    """Strip account path components from HTTPX's otherwise automatic request log."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = re.sub(r"(/v3/accounts/)[^/?\s]+", r"\1<redacted>", record.getMessage())
        record.args = ()
        return True


def summarize_account_candle(
    response: httpx.Response,
    start: datetime,
) -> AccountCandleObservation:
    """Validate one raw minute and project only non-price observation metadata."""
    if response.status_code != 200:
        return AccountCandleObservation("HTTP_ERROR", response.status_code)
    try:
        payload = _AccountEnvelope.model_validate_json(response.content)
        if not payload.candles:
            return AccountCandleObservation("EMPTY", 200, 0, 0)
        if len(payload.candles) != 1:
            return AccountCandleObservation("SCHEMA_ERROR", 200)
        candle = payload.candles[0]
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)", candle.time
        ):
            return AccountCandleObservation("SCHEMA_ERROR", 200)
        timestamp = datetime.fromisoformat(candle.time.replace("Z", "+00:00"))
        if timestamp != start:
            return AccountCandleObservation("SCHEMA_ERROR", 200)
        rows = _bid_ask_rows(
            payload, start, start + timedelta(minutes=1) - timedelta(microseconds=1)
        )
    except (PayloadValidationError, ValidationError, ValueError):
        return AccountCandleObservation("SCHEMA_ERROR", 200)
    return AccountCandleObservation(
        "OBSERVED" if candle.complete else "INCOMPLETE",
        200,
        1,
        len(rows),
        (timestamp,),
    )


def probe_account_candle(
    client: httpx.Client,
    base_url: str,
    account_id: str,
) -> AccountCandleObservation:
    """Make one fixed-scope request; execution callers must supply one-use controls."""
    if base_url not in ("https://api-fxpractice.oanda.com", "https://api-fxtrade.oanda.com"):
        raise ValidationError("Unapproved OANDA host")
    if not re.fullmatch(r"[A-Za-z0-9-]+", account_id):
        raise ValidationError("Invalid OANDA account identifier")
    url = f"{base_url}/v3/accounts/{quote(account_id, safe='')}/instruments/EUR_USD/candles"
    logger = logging.getLogger("httpx")
    redactor = _AccountPathRedactor()
    logger.addFilter(redactor)
    try:
        response = client.get(
            url,
            params={
                "from": "2024-01-03T00:00:00Z",
                "to": "2024-01-03T00:00:59.999999Z",
                "price": "BA",
                "granularity": "M1",
                "units": "1",
                "smooth": "false",
                "includeFirst": "true",
            },
            timeout=30.0,
            follow_redirects=False,
        )
    except httpx.RequestError:
        return AccountCandleObservation("TRANSPORT_ERROR", None)
    finally:
        logger.removeFilter(redactor)
    return summarize_account_candle(response, PROBE_START)

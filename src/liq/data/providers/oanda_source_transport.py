"""Narrow account-scoped source-verification transport, never execution permission."""

import logging
import re
from collections.abc import Mapping
from datetime import date
from typing import TYPE_CHECKING, Final
from urllib.parse import quote

import httpx

from liq.data.exceptions import ValidationError
from liq.data.oanda_instrument_metadata import (
    MATRIX_SYMBOLS,
    InstrumentMetadataObservation,
    summarize_instruments,
)
from liq.data.oanda_source_metadata import CandleDay, CandleDayObservation, summarize_candle_day
from liq.data.providers.oanda_account_probe import _AccountPathRedactor

if TYPE_CHECKING:
    from liq.data.providers.oanda import OandaProvider

MATRIX_DAYS: Final = (
    date(2020, 1, 3),
    date(2024, 1, 3),
    date(2024, 1, 31),
    date(2024, 3, 15),
    date(2024, 7, 3),
    date(2024, 12, 30),
)


def validate_source_day(scope: CandleDay) -> None:
    """Refuse any case outside the independently reviewed source-only matrix."""
    if scope.symbol not in MATRIX_SYMBOLS or scope.day not in MATRIX_DAYS:
        raise ValidationError("Unapproved OANDA source-summary case")


def _account_get(
    provider: "OandaProvider", resource: str, params: Mapping[str, str]
) -> httpx.Response | None:
    """Make one non-redirecting account GET; callers fix resource and parameters."""
    if provider._base_url not in (
        "https://api-fxpractice.oanda.com",
        "https://api-fxtrade.oanda.com",
    ) or not re.fullmatch(r"[A-Za-z0-9-]+", provider._account_id):
        raise ValidationError("Invalid OANDA source-verification endpoint identity")
    url = f"{provider._base_url}/v3/accounts/{quote(provider._account_id, safe='')}/{resource}"
    logger = logging.getLogger("httpx")
    redactor = _AccountPathRedactor()
    logger.addFilter(redactor)
    try:
        return provider._get_client().get(url, params=params, timeout=30.0, follow_redirects=False)
    except httpx.RequestError:
        return None
    finally:
        logger.removeFilter(redactor)


def probe_instrument_metadata(provider: "OandaProvider") -> InstrumentMetadataObservation:
    """Request only the fixed seven-symbol membership and native-unit projection."""
    response = _account_get(provider, "instruments", {"instruments": ",".join(MATRIX_SYMBOLS)})
    if response is None:
        return InstrumentMetadataObservation(category="TRANSPORT_ERROR", http_status=None)
    return summarize_instruments(response)


def probe_source_day(provider: "OandaProvider", scope: CandleDay) -> CandleDayObservation:
    """Fetch one exact UTC day, with no pagination, replacement date or fallback route."""
    validate_source_day(scope)
    day = scope.day.isoformat()
    response = _account_get(
        provider,
        f"instruments/{scope.symbol}/candles",
        {
            "from": f"{day}T00:00:00Z",
            "to": f"{day}T23:59:59.999999Z",
            "price": "BA",
            "granularity": "M1",
            "units": "1",
            "smooth": "false",
            "includeFirst": "true",
        },
    )
    if response is None:
        return CandleDayObservation("TRANSPORT_ERROR", None)
    if response.status_code != 200:
        return CandleDayObservation("HTTP_ERROR", response.status_code)
    try:
        summary = summarize_candle_day(response, scope)
    except ValidationError:
        return CandleDayObservation("SCHEMA_ERROR", 200)
    return CandleDayObservation("OBSERVED", 200, summary)

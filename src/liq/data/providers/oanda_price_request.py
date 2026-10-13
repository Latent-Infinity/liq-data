"""Private snapshot-bound structural transport shared by concrete source wrappers."""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import httpx
from pydantic import ValidationError

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_cohort_capture import validate_cohort_capture_window
from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.oanda_source_envelope import (
    CaptureSecrets,
    SourceEnvelopeRefused,
    _parse_source_envelope,
)
from liq.data.providers.oanda_account_probe import _AccountPathRedactor
from liq.data.providers.oanda_price_capture import PriceCaptureFailure, PriceCapturePolicyRefused
from liq.data.providers.oanda_source_body import (
    SourceBody,
    SourceBodyFailure,
    read_source_body,
)

if TYPE_CHECKING:
    from liq.data.providers.oanda import OandaProvider


@dataclass(frozen=True, slots=True)
class _SourcePriceRequest:
    """A concrete fixed-case declaration, never caller-selected query parameters."""

    window: SourcePriceWindow | CohortSourceWindow


def _stream_source_prices(
    provider: "OandaProvider", request: _SourcePriceRequest
) -> SourceBody | PriceCaptureFailure:
    if type(request) is not _SourcePriceRequest or type(request.window) not in (
        SourcePriceWindow,
        CohortSourceWindow,
    ):
        return PriceCaptureFailure("WINDOW_REFUSED", None, 0)
    window: SourcePriceWindow | CohortSourceWindow
    try:
        match request.window:
            case SourcePriceWindow() as declared:
                window = SourcePriceWindow.model_validate(declared.model_dump(warnings=False))
            case CohortSourceWindow() as declared:
                window = validate_cohort_capture_window(declared)
            case unreachable:
                assert_never(unreachable)
    except (ValidationError, PriceCapturePolicyRefused):
        return PriceCaptureFailure("WINDOW_REFUSED", None, 0)
    symbol = window.symbol
    parameters = tuple(window.query().items())
    if (
        provider._environment != "live"
        or provider._base_url != "https://api-fxtrade.oanda.com"
        or re.fullmatch(r"[A-Za-z0-9-]+", provider._account_id) is None
    ):
        return PriceCaptureFailure("ENDPOINT_REFUSED", None, 0)
    try:
        secrets = CaptureSecrets(provider._account_id, provider._api_key)
        secrets.api_token.encode("ascii")
    except SourceEnvelopeRefused as error:
        return PriceCaptureFailure(error.reason, None, 0)
    except UnicodeEncodeError:
        return PriceCaptureFailure("INVALID_SECRETS", None, 0)
    url = f"{provider._base_url}/v3/accounts/{provider._account_id}/instruments/{symbol}/candles"
    logger = logging.getLogger("httpx")
    redactor = _AccountPathRedactor()
    logger.addFilter(redactor)
    try:
        with provider._get_client().stream(
            "GET",
            url,
            params=parameters,
            headers={"Accept-Encoding": "identity"},
            timeout=30.0,
            follow_redirects=False,
        ) as response:
            result = read_source_body(response)
            match result:
                case SourceBodyFailure(reason=reason, http_status=status, received_bytes=count):
                    return PriceCaptureFailure(reason, status, count)
                case SourceBody() as body:
                    try:
                        _parse_source_envelope(
                            body.body, symbol, (secrets.account_id, secrets.api_token)
                        )
                    except SourceEnvelopeRefused as error:
                        return PriceCaptureFailure(error.reason, 200, len(body.body))
                    return body
                case unreachable:
                    assert_never(unreachable)
    except (httpx.RequestError, httpx.StreamError):
        return PriceCaptureFailure("TRANSPORT_ERROR", None, 0)
    finally:
        logger.removeFilter(redactor)

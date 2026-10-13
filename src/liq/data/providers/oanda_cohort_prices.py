"""Concrete cohort transport wrapper; no legacy scope or profile selection."""

from typing import TYPE_CHECKING

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_cohort_capture import validate_cohort_capture_window
from liq.data.providers.oanda_price_capture import PriceCaptureFailure, PriceCapturePolicyRefused
from liq.data.providers.oanda_price_request import _SourcePriceRequest, _stream_source_prices
from liq.data.providers.oanda_source_body import SourceBody

if TYPE_CHECKING:
    from liq.data.providers.oanda import OandaProvider


def capture_cohort_source_prices(
    provider: "OandaProvider", scope: CohortSourceWindow
) -> SourceBody | PriceCaptureFailure:
    """Snapshot one fixed case before client resolution; return no admitted rows."""
    try:
        window = validate_cohort_capture_window(scope)
    except PriceCapturePolicyRefused:
        return PriceCaptureFailure("WINDOW_REFUSED", None, 0)
    request = _SourcePriceRequest(window)
    return _stream_source_prices(provider, request)

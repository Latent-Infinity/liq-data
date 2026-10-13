"""Closed cohort ingress; the actual research guard precedes provider resolution."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import ValidationError

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.providers.oanda_account_probe import PROBE_ARM
from liq.data.providers.oanda_price_capture import PriceCaptureFailure, PriceCapturePolicyRefused
from liq.data.providers.oanda_source_body import SourceBody

if TYPE_CHECKING:
    from liq.data.service import DataService


@runtime_checkable
class CohortSourceCaptureProvider(Protocol):
    """Optional cohort transport capability, never research or one-use authority."""

    def capture_cohort_source_prices(
        self, scope: CohortSourceWindow
    ) -> SourceBody | PriceCaptureFailure:
        """Return structural quarantine only, not native-quality admission."""
        ...


def validate_cohort_capture_window(scope: CohortSourceWindow) -> CohortSourceWindow:
    """Refuse alternate types and forged fields without exposing parser inputs."""
    if type(scope) is not CohortSourceWindow:
        raise PriceCapturePolicyRefused("WINDOW_REFUSED")
    try:
        return CohortSourceWindow.model_validate(scope.model_dump(warnings=False))
    except ValidationError:
        raise PriceCapturePolicyRefused("WINDOW_REFUSED") from None


def capture_guarded_cohort_prices(
    service: "DataService", scope: CohortSourceWindow, *, purpose: str, arm_id: str
) -> SourceBody | PriceCaptureFailure:
    """Use the facade's explicit audit declaration; no store or generic bypass."""
    if purpose != "discovery" or arm_id != PROBE_ARM:
        raise PriceCapturePolicyRefused("DECLARATION_REFUSED")
    window = validate_cohort_capture_window(scope)
    ingress_window = window.model_copy()
    service._prepare_oanda_cohort_capture(ingress_window)
    if validate_cohort_capture_window(ingress_window) != window:
        raise PriceCapturePolicyRefused("WINDOW_REFUSED")
    if (
        service.lockbox_guard.classify_read(
            "oanda", window.symbol, asset_class="forex", data_kind="bars"
        )
        != "oanda_fx"
    ):
        raise PriceCapturePolicyRefused("CLASSIFICATION_REFUSED")
    service._guard_research_read(
        "oanda",
        window.symbol,
        window.day,
        window.day,
        purpose=purpose,
        arm_id=arm_id,
        final_portfolio_review=False,
        asset_class="forex",
        timeframe="1m",
        data_kind="bars",
    )
    if validate_cohort_capture_window(ingress_window) != window:
        raise PriceCapturePolicyRefused("WINDOW_REFUSED")
    selector_window = ingress_window
    source = service._get_oanda_cohort_capture_provider(selector_window)
    if validate_cohort_capture_window(selector_window) != window:
        raise PriceCapturePolicyRefused("WINDOW_REFUSED")
    if not isinstance(source, CohortSourceCaptureProvider):
        raise PriceCapturePolicyRefused("PROVIDER_UNSUPPORTED")
    return source.capture_cohort_source_prices(window)

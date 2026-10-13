"""Exact BA source transport; callers need guarded, one-use capture authority."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.oanda_source_envelope import EnvelopeFailureReason
from liq.data.providers.oanda_source_body import (
    BodyFailureReason,
    SourceBody,
)

if TYPE_CHECKING:
    from liq.data.providers.oanda import OandaProvider

CaptureFailureReason = (
    BodyFailureReason | EnvelopeFailureReason | Literal["ENDPOINT_REFUSED", "WINDOW_REFUSED"]
)


@dataclass(frozen=True, slots=True)
class PriceCaptureFailure:
    """Safe request disposition without raw bytes, credentials or exception text."""

    reason: CaptureFailureReason
    http_status: int | None
    received_bytes: int


class PriceCapturePolicyRefused(ValueError):
    """A source capture declaration or required guarded capability is unavailable."""

    def __init__(
        self,
        reason: Literal[
            "DECLARATION_REFUSED",
            "CLASSIFICATION_REFUSED",
            "PROVIDER_UNSUPPORTED",
            "WINDOW_REFUSED",
        ],
    ) -> None:
        self.reason = reason
        super().__init__(reason)


def capture_source_prices(
    provider: "OandaProvider", scope: SourcePriceWindow
) -> SourceBody | PriceCaptureFailure:
    """Make one streaming GET and return only structurally quarantined bytes.

    This transport capability is not a research entry point or permission to
    publish. The collector must call the guarded DataService facade and impose
    its separately tested hard request deadline before transport through close.
    """
    from liq.data.providers.oanda_price_request import _SourcePriceRequest, _stream_source_prices

    try:
        window = SourcePriceWindow.model_validate(scope.model_dump())
    except ValidationError:
        return PriceCaptureFailure("WINDOW_REFUSED", None, 0)
    request = _SourcePriceRequest(window)
    return _stream_source_prices(provider, request)

"""Finite current-vintage source for provider-adjusted arithmetic only."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from liq.data.alpaca_announcement_source import (
    ANNOUNCEMENT_END,
    AlpacaAnnouncementDailyRequest,
    AnnouncementAccession,
)
from liq.data.alpaca_source import AlpacaCaptureRefused


class AlpacaEventBasisDailyRequest(BaseModel):
    """Sixteen paired raw/all slots; neither mode establishes total return or PIT.

    Both modes use the same frozen event/session map as the raw announcement
    request. The mapping date identifies symbols, not the data acquisition clock.
    Exact two-row completeness is enforced separately before numerical use.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )
    symbol: Literal["KHC", "SPY"]
    accession: AnnouncementAccession
    adjustment: Literal["raw", "all"]
    asof: date = ANNOUNCEMENT_END
    timeframe: Literal["1Day"] = "1Day"
    feed: Literal["sip"] = "sip"
    currency: Literal["USD"] = "USD"
    limit: Literal[4] = 4
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[131072] = 131072

    @property
    def start(self) -> datetime:
        return AlpacaAnnouncementDailyRequest(symbol=self.symbol, accession=self.accession).start

    @property
    def end(self) -> datetime:
        return AlpacaAnnouncementDailyRequest(symbol=self.symbol, accession=self.accession).end

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaEventBasisDailyRequest":
        if self.asof != ANNOUNCEMENT_END:
            raise AlpacaCaptureRefused("event_basis_daily_scope")
        return self


def validate_event_basis_request(
    request: AlpacaEventBasisDailyRequest,
) -> AlpacaEventBasisDailyRequest:
    """Refuse subclass behavior and revalidate constructed or copied instances."""
    if type(request) is not AlpacaEventBasisDailyRequest:
        raise AlpacaCaptureRefused("event_basis_request_type")
    return AlpacaEventBasisDailyRequest.model_validate(request)

"""Finite current-vintage requests; calendar confirmation precedes price access."""

from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from types import MappingProxyType
from typing import Final, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, model_validator

from liq.data.alpaca_source import AlpacaCaptureRefused

AnnouncementAccession = Literal[
    "0001637459-22-000056",
    "0001637459-22-000156",
    "0001637459-22-000186",
    "0001637459-23-000005",
]
ANNOUNCEMENT_START: Final = date(2022, 4, 25)
ANNOUNCEMENT_END: Final = date(2023, 2, 15)
ANNOUNCEMENT_DATE_PAIRS: Final[Mapping[AnnouncementAccession, tuple[date, date]]] = (
    MappingProxyType(
        {
            "0001637459-22-000056": (date(2022, 4, 26), date(2022, 4, 27)),
            "0001637459-22-000156": (date(2022, 7, 26), date(2022, 7, 27)),
            "0001637459-22-000186": (date(2022, 10, 25), date(2022, 10, 26)),
            "0001637459-23-000005": (date(2023, 2, 14), date(2023, 2, 15)),
        }
    )
)


class AlpacaAnnouncementCalendarRequest(BaseModel):
    """Proposed session coverage, without asserting any calendar observation."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )
    start: date = ANNOUNCEMENT_START
    end: date = ANNOUNCEMENT_END
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[131072] = 131072
    max_records: Literal[300] = 300

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaAnnouncementCalendarRequest":
        if self.start != ANNOUNCEMENT_START or self.end != ANNOUNCEMENT_END:
            raise AlpacaCaptureRefused("announcement_calendar_scope")
        return self


class AlpacaAnnouncementDailyRequest(BaseModel):
    """Eight proposed slots; entity lookup date does not establish source vintage.

    Query bounds are native day-label midnights, not approximated session clocks.
    The calendar must establish the corresponding strict-before/after sessions.
    A response must contain exactly the two expected bars, without selecting extras.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )
    symbol: Literal["KHC", "SPY"]
    accession: AnnouncementAccession
    asof: date = ANNOUNCEMENT_END
    timeframe: Literal["1Day"] = "1Day"
    adjustment: Literal["raw"] = "raw"
    feed: Literal["sip"] = "sip"
    currency: Literal["USD"] = "USD"
    limit: Literal[4] = 4
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[131072] = 131072

    @property
    def start(self) -> datetime:
        return datetime.combine(
            ANNOUNCEMENT_DATE_PAIRS[self.accession][0],
            time.min,
            tzinfo=ZoneInfo("America/New_York"),
        ).astimezone(UTC)

    @property
    def end(self) -> datetime:
        return datetime.combine(
            ANNOUNCEMENT_DATE_PAIRS[self.accession][1],
            time.min,
            tzinfo=ZoneInfo("America/New_York"),
        ).astimezone(UTC)

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaAnnouncementDailyRequest":
        if self.asof != ANNOUNCEMENT_END:
            raise AlpacaCaptureRefused("announcement_daily_scope")
        return self


def validate_announcement_request(
    request: AlpacaAnnouncementCalendarRequest | AlpacaAnnouncementDailyRequest,
) -> AlpacaAnnouncementCalendarRequest | AlpacaAnnouncementDailyRequest:
    """Adopt only trusted concrete models, including at direct provider entry."""
    if type(request) not in (AlpacaAnnouncementCalendarRequest, AlpacaAnnouncementDailyRequest):
        raise AlpacaCaptureRefused("announcement_request_type")
    return type(request).model_validate(request)

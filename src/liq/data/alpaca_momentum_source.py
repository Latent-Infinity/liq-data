"""Fixed current-vintage KHC request scope, separate from consumed captures."""

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, field_validator, model_validator

from liq.data.alpaca_source import AlpacaCaptureRefused

MOMENTUM_START = date(2022, 4, 1)
MOMENTUM_END = date(2023, 3, 31)
MOMENTUM_MONTHS = tuple((2022, month) for month in range(4, 13)) + tuple(
    (2023, month) for month in range(1, 4)
)


class AlpacaMomentumCalendarRequest(BaseModel):
    """One provider-declared trading-calendar page, without historical admission."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    start: date = MOMENTUM_START
    end: date = MOMENTUM_END
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[131072] = 131072

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaMomentumCalendarRequest":
        if self.start != MOMENTUM_START or self.end != MOMENTUM_END:
            raise AlpacaCaptureRefused("momentum_calendar_scope")
        return self


class AlpacaMomentumDailyRequest(BaseModel):
    """One all-adjusted annual page; asof is symbol mapping, never data vintage."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, revalidate_instances="always"
    )

    symbol: Literal["KHC"] = "KHC"
    start: AwareDatetime = datetime(2022, 4, 1, 4, tzinfo=UTC)
    end: AwareDatetime = datetime(2023, 3, 31, 4, tzinfo=UTC)
    asof: date = MOMENTUM_END
    timeframe: Literal["1Day"] = "1Day"
    adjustment: Literal["all"] = "all"
    feed: Literal["sip"] = "sip"
    currency: Literal["USD"] = "USD"
    limit: Literal[400] = 400
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[131072] = 131072

    @field_validator("start", "end")
    @classmethod
    def utc_bounds(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaMomentumDailyRequest":
        if (
            self.start != datetime(2022, 4, 1, 4, tzinfo=UTC)
            or self.end != datetime(2023, 3, 31, 4, tzinfo=UTC)
            or self.asof != MOMENTUM_END
        ):
            raise AlpacaCaptureRefused("momentum_daily_scope")
        return self

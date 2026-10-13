from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)

SOURCE_PAGE_TIMEOUT_SECONDS = 20.0


class AlpacaCalendarRequest(BaseModel):
    """One retail calendar reference capture; no historical publication admission."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    start: date = date(2023, 1, 26)
    end: date = date(2023, 4, 24)
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[1048576] = 1048576

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaCalendarRequest":
        if self.start != date(2023, 1, 26) or self.end != date(2023, 4, 24):
            raise ValueError("calendar reference is outside its fixed interval")
        return self


class AlpacaDailyFixtureRequest(BaseModel):
    """One fixed present-vintage daily arithmetic request, independent of minute scope."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbol: Literal["KHC", "SPY"]
    start: AwareDatetime = datetime(2023, 1, 26, 5, tzinfo=UTC)
    end: AwareDatetime = datetime(2023, 4, 24, 4, tzinfo=UTC)
    asof: date = date(2023, 4, 24)
    timeframe: Literal["1Day"] = "1Day"
    adjustment: Literal["all"] = "all"
    feed: Literal["sip"] = "sip"
    currency: Literal["USD"] = "USD"
    limit: Literal[1000] = 1000
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[1048576] = 1048576

    @field_validator("start", "end")
    @classmethod
    def utc_bounds(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaDailyFixtureRequest":
        if (
            self.start.astimezone(UTC) != datetime(2023, 1, 26, 5, tzinfo=UTC)
            or self.end.astimezone(UTC) != datetime(2023, 4, 24, 4, tzinfo=UTC)
            or self.asof != date(2023, 4, 24)
        ):
            raise ValueError("daily request is outside its fixed arithmetic fixture scope")
        return self


_ACTION_WINDOWS = (
    (("LEH", "LEHMQ"), date(2008, 1, 1), date(2008, 12, 31)),
    (("PCLN", "BKNG"), date(2009, 1, 1), date(2009, 12, 31)),
    (("PCLN", "BKNG"), date(2018, 1, 1), date(2018, 12, 31)),
)


class AlpacaCaptureRefused(ValueError):
    def __init__(self, reason: str, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.status_code = status_code


class AlpacaCorporateActionsRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbols: tuple[StrictStr, StrictStr]
    start: date
    end: date
    data_quality: Literal["complete"] = "complete"
    limit: Literal[1000] = 1000
    max_pages: Annotated[int, Field(strict=True, ge=1, le=2)] = 2
    max_response_bytes: Annotated[int, Field(strict=True, ge=1, le=1048576)] = 1048576

    @field_validator("start", "end", mode="before")
    @classmethod
    def strict_calendar_date(cls, value: object) -> object:
        if type(value) is not date:
            raise ValueError("action bounds must be calendar dates")
        return value

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaCorporateActionsRequest":
        if (self.symbols, self.start, self.end) not in _ACTION_WINDOWS:
            raise ValueError("corporate-action request is outside approved characterization scope")
        if any(
            symbol != symbol.strip() or symbol.upper() in {"ALL_SYMBOLS", "*"}
            for symbol in self.symbols
        ):
            raise ValueError("corporate-action request requires literal symbols")
        return self


class AlpacaBarsRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    symbol: Literal["KHC", "SPY"]
    start: AwareDatetime
    end: AwareDatetime
    asof: date
    timeframe: Literal["1Min"] = "1Min"
    adjustment: Literal["raw", "all"]
    feed: Literal["sip"]
    currency: Literal["USD"] = "USD"
    limit: Literal[1000] = 1000
    max_pages: Annotated[int, Field(strict=True, ge=1, le=2)] = 2
    max_response_bytes: Annotated[int, Field(strict=True, ge=1, le=1048576)] = 1048576

    @field_validator("asof", mode="before")
    @classmethod
    def strict_asof_date(cls, value: object) -> object:
        if type(value) is not date:
            raise ValueError("asof must be a calendar date")
        return value

    @model_validator(mode="after")
    def exact_scope(self) -> "AlpacaBarsRequest":
        if self.start.astimezone(UTC) >= self.end.astimezone(UTC):
            raise ValueError("bar interval start must precede end")
        if (
            self.start.astimezone(UTC) != datetime(2023, 4, 21, 13, 30, tzinfo=UTC)
            or self.end.astimezone(UTC) != datetime(2023, 4, 24, 19, 59, tzinfo=UTC)
            or self.asof != date(2023, 4, 24)
        ):
            raise ValueError("bar request is outside approved KHC/SPY fixture window")
        return self

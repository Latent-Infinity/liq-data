"""Exact source-price calibration windows; this contract grants no transport permission."""

from datetime import UTC, date, datetime, time
from typing import Literal, TypedDict
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, model_validator

from liq.data.oanda_instrument_metadata import MATRIX_SYMBOLS
from liq.data.providers.oanda_source_transport import MATRIX_DAYS

PriceCaptureQuery = TypedDict(
    "PriceCaptureQuery",
    {
        "from": str,
        "to": str,
        "price": Literal["BA"],
        "granularity": Literal["M1"],
        "units": Literal["1"],
        "smooth": Literal["false"],
        "includeFirst": Literal["true"],
    },
)


class SourceWindowRefused(ValueError):
    """The requested pair/day is outside the reviewed source-capture universe."""


class SourcePriceWindow(BaseModel):
    """Parse a declared pair/day before deriving its immutable approved clock range."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    symbol: str
    day: date

    @model_validator(mode="after")
    def approved_case(self) -> "SourcePriceWindow":
        if self.symbol not in MATRIX_SYMBOLS or self.day not in MATRIX_DAYS:
            raise SourceWindowRefused
        return self

    def query(self) -> PriceCaptureQuery:
        """Derive the explicit native-candle range, never a caller-supplied override."""
        lower = datetime.combine(self.day, time(13, 39), ZoneInfo("Europe/London"))
        upper = datetime.combine(self.day, time(16, 29, 59, 999999), ZoneInfo("Europe/London"))
        return {
            "from": lower.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "to": upper.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "price": "BA",
            "granularity": "M1",
            "units": "1",
            "smooth": "false",
            "includeFirst": "true",
        }

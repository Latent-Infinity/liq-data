"""Development-cohort source windows; metadata grants no transport or scientific use."""

from datetime import date
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from liq.data.fx_fix_source_calendar import fx_fix_source_day
from liq.data.oanda_instrument_metadata import MATRIX_SYMBOLS
from liq.data.oanda_source_capture_contract import PriceCaptureQuery

FX_FIX_COHORT_EVALUATION_DAYS: Final = frozenset(
    (
        date(2024, 1, 31),
        date(2024, 3, 28),
        date(2024, 4, 2),
        date(2024, 7, 3),
        date(2024, 10, 25),
        date(2024, 10, 28),
    )
)


class CohortWindowRefused(ValueError):
    """A development source-window request is outside its fixed scope."""


class CohortSourceWindow(BaseModel):
    """Closed, immutable scope distinct from the legacy source-QA window.

    Weekdays are availability candidates, including holidays, not proof of WMR
    publication. The required scope tag prevents legacy payload substitution; it
    is not authority to select a transport profile or contact a provider.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    scope: Literal["london-fx-fix-development-source/v1"]
    symbol: str
    day: date
    publication_status: Literal["unverified"] = "unverified"

    @model_validator(mode="after")
    def fixed_case(self) -> "CohortSourceWindow":
        training_day = self.day.year == 2023 and self.day.weekday() < 5
        if self.symbol not in MATRIX_SYMBOLS or not (
            training_day or self.day in FX_FIX_COHORT_EVALUATION_DAYS
        ):
            raise CohortWindowRefused
        return self

    def query(self) -> PriceCaptureQuery:
        """Revalidate scope before deriving the shared London BA/M1 calendar clock."""
        window = CohortSourceWindow.model_validate(self.model_dump(warnings=False))
        calendar = fx_fix_source_day(window.day)
        if calendar is None:
            raise CohortWindowRefused
        return {
            "from": calendar.start_utc.isoformat().replace("+00:00", "Z"),
            "to": calendar.end_utc.isoformat().replace("+00:00", "Z"),
            "price": "BA",
            "granularity": "M1",
            "units": "1",
            "smooth": "false",
            "includeFirst": "true",
        }

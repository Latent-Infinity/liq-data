"""Current OANDA instrument units and source-verification request contracts."""

from datetime import UTC, datetime
from typing import Annotated, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator
from pydantic import ValidationError as PayloadValidationError

MATRIX_SYMBOLS: Final = (
    "AUD_USD",
    "EUR_USD",
    "GBP_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
)
Sha256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class InstrumentMetadataRequest(BaseModel):
    """Declared metadata scope and provenance; not a grant of transport permission."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    attempt_id: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{32}$")]
    source_sha256: Sha256
    config_sha256: Sha256
    permit_sha256: Sha256
    purpose: StrictStr | None
    arm_id: StrictStr
    symbols: tuple[StrictStr, ...]
    final_portfolio_review: bool = False


class InstrumentUnits(BaseModel):
    """Native instrument metadata; pip location is an exponent in quote-price units."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    name: StrictStr
    type: Literal["CURRENCY"]
    pip_location: StrictInt = Field(alias="pipLocation")
    display_precision: Annotated[StrictInt, Field(ge=0)] = Field(alias="displayPrecision")

    @model_validator(mode="after")
    def representable_pip(self) -> "InstrumentUnits":
        if self.pip_location < -self.display_precision:
            raise ValueError("Pip location exceeds displayed decimal precision")
        return self


class InstrumentMetadataObservation(BaseModel):
    """Minimized result; observed_at is local receipt/parse time, never provider as-of."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    category: Literal["OBSERVED", "HTTP_ERROR", "TRANSPORT_ERROR", "SCHEMA_ERROR"]
    http_status: int | None
    instruments: tuple[InstrumentUnits, ...] = ()
    missing_symbols: tuple[str, ...] = ()
    observed_at: datetime | None = None


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    instruments: tuple[InstrumentUnits, ...]


def summarize_instruments(response: httpx.Response) -> InstrumentMetadataObservation:
    """Reject unknown/duplicate symbols and malformed units without retaining bodies."""
    if response.status_code != 200:
        return InstrumentMetadataObservation(
            category="HTTP_ERROR", http_status=response.status_code
        )
    try:
        payload = _Envelope.model_validate_json(response.content)
    except PayloadValidationError:
        return InstrumentMetadataObservation(category="SCHEMA_ERROR", http_status=200)
    names = tuple(item.name for item in payload.instruments)
    if len(set(names)) != len(names) or not set(names) <= set(MATRIX_SYMBOLS):
        return InstrumentMetadataObservation(category="SCHEMA_ERROR", http_status=200)
    return InstrumentMetadataObservation(
        category="OBSERVED",
        http_status=200,
        instruments=tuple(sorted(payload.instruments, key=lambda item: item.name)),
        missing_symbols=tuple(symbol for symbol in MATRIX_SYMBOLS if symbol not in names),
        observed_at=datetime.now(UTC),
    )

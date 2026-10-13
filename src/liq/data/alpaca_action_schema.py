from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from liq.data.alpaca_source import AlpacaCaptureRefused
from liq.data.lockbox import FoldWindows, LockboxGuard, LockboxLedger

ACTION_SCHEMA_DATASET: Final = "alpaca_ranker_action_schema"
ACTION_SCHEMA_LEDGER: Final = LockboxLedger(
    version="classical_ranker_action_schema_v1.0",
    datasets=MappingProxyType(
        {
            ACTION_SCHEMA_DATASET: FoldWindows(
                characterization=(date(2022, 1, 1), date(2023, 3, 31)),
                allowed_purposes=frozenset(),
            )
        }
    ),
)


class AlpacaActionSchemaRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    symbols: tuple[Literal["KHC"], Literal["SPY"]] = ("KHC", "SPY")
    start: date = date(2022, 1, 1)
    end: date = date(2023, 3, 31)
    data_quality: Literal["complete"] = "complete"
    limit: Literal[1000] = 1000
    max_pages: Literal[1] = 1
    max_response_bytes: Literal[1048576] = 1048576

    @model_validator(mode="after")
    def exact_dates(self) -> "AlpacaActionSchemaRequest":
        if self.start != date(2022, 1, 1) or self.end != date(2023, 3, 31):
            raise AlpacaCaptureRefused("action_schema_scope")
        return self


def validate_action_schema_request(request: AlpacaActionSchemaRequest) -> AlpacaActionSchemaRequest:
    if type(request) is not AlpacaActionSchemaRequest:
        raise AlpacaCaptureRefused("action_schema_request_type")
    try:
        return AlpacaActionSchemaRequest.model_validate(request.model_dump())
    except ValidationError as exc:
        raise AlpacaCaptureRefused("action_schema_scope") from exc


def guard_action_schema(
    request: AlpacaActionSchemaRequest,
    usage_log_path: Path,
    *,
    purpose: str = "dev_smoke",
    arm_id: str = "classical_cross_sectional_ranker",
    final_portfolio_review: bool = False,
) -> None:
    request = validate_action_schema_request(request)
    if (
        purpose != "dev_smoke"
        or arm_id != "classical_cross_sectional_ranker"
        or final_portfolio_review
    ):
        raise AlpacaCaptureRefused("action_schema_declaration")
    LockboxGuard(usage_log_path, ledger=ACTION_SCHEMA_LEDGER).assert_period_allowed(
        ACTION_SCHEMA_DATASET,
        request.start,
        request.end,
        purpose=purpose,
        arm_id=arm_id,
        final_portfolio_review=False,
    )

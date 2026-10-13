"""Typed metadata for three reads of one retained engineering fixture."""

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Invocation = Literal["reference", "focused", "full-feature"]


class MomentumNumericalRefused(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def parse_invocation(value: str) -> Invocation:
    match value:
        case "reference":
            return "reference"
        case "focused":
            return "focused"
        case "full-feature":
            return "full-feature"
        case _:
            raise MomentumNumericalRefused("numerical_invocation")


class MetadataModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", revalidate_instances="always"
    )


class MomentumNumericalRead(MetadataModel):
    """A fixed declaration; this is not an access grant."""

    symbol: Literal["KHC"] = "KHC"
    start: date = date(2022, 4, 1)
    end: date = date(2023, 3, 31)
    dataset: Literal["alpaca_ranker_momentum_numerical_fixture"] = (
        "alpaca_ranker_momentum_numerical_fixture"
    )
    purpose: Literal["dev_smoke"] = "dev_smoke"
    arm_id: Literal["classical_cross_sectional_ranker_momentum_numerical_fixture"] = (
        "classical_cross_sectional_ranker_momentum_numerical_fixture"
    )
    final_portfolio_review: Literal[False] = False
    decoded_daily_records: Literal[251] = 251
    selected_endpoints: Literal[12] = 12
    max_numerical_reads: Literal[3] = 3

    @model_validator(mode="after")
    def exact_period(self) -> "MomentumNumericalRead":
        if self.start != date(2022, 4, 1) or self.end != date(2023, 3, 31):
            raise MomentumNumericalRefused("numerical_period")
        return self


class FilePin(MetadataModel):
    path: StrictStr
    sha256: Digest
    bytes: Annotated[int, Field(ge=0)]


class ImportPin(FilePin):
    module: StrictStr


class NumericalRuntime(MetadataModel):
    cwd: StrictStr
    interpreter: FilePin
    interpreter_invocation: StrictStr
    timezone: FilePin
    timezone_search_path: tuple[StrictStr, ...]
    code_pins: tuple[FilePin, ...]
    origins: tuple[ImportPin, ...]
    bootstrap_modules: tuple[StrictStr, ...]
    package_versions: tuple[tuple[StrictStr, StrictStr], ...]


class NumericalContract(MetadataModel):
    version: Literal["khc_momentum_numerical_v1"] = "khc_momentum_numerical_v1"
    request: MomentumNumericalRead = MomentumNumericalRead()
    fixture_root: StrictStr
    usage_root: StrictStr
    source_receipt: FilePin
    source_lineage: tuple[FilePin, ...]
    runtime: FilePin
    reference_path: StrictStr
    original_test: FilePin
    current_test: StrictStr
    reference_placeholder: Literal[
        "0000000000000000000000000000000000000000000000000000000000000000"
    ] = "0000000000000000000000000000000000000000000000000000000000000000"
    invocations: tuple[Literal["reference"], Literal["focused"], Literal["full-feature"]] = (
        "reference",
        "focused",
        "full-feature",
    )
    decimal_precision: Literal[50] = 50
    absolute_fractional_tolerance: Literal["1e-12"] = "1e-12"
    provider_requests: Literal[0] = 0
    additional_spending: Literal[False] = False
    real_orders: Literal[False] = False


class NumericalAuthority(MetadataModel):
    version: Literal["khc_momentum_numerical_authority_v1"] = "khc_momentum_numerical_authority_v1"
    decision: Literal["APPROVE"]
    contract_sha256: Digest
    oracle_decision: FilePin
    max_numerical_reads: Literal[3] = 3
    max_decoded_daily_records: Literal[753] = 753


class CapturedFile(MetadataModel):
    name: StrictStr
    sha256: Digest
    bytes: Annotated[int, Field(ge=0, le=131072)]
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


class CaptureClocks(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    status: Literal["SOURCE_CHARACTERIZATION_ONLY"]
    calendar_captured_at: StrictStr
    daily_captured_at: StrictStr
    calendar_records: Literal[251]
    daily_records: Literal[251]
    selected_endpoints: Literal[12]


class SourceReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    receipt: CaptureClocks
    source_witnesses: tuple[CapturedFile, ...]


class ReadReservation(MetadataModel):
    contract_sha256: Digest
    authority_sha256: Digest
    invocation: Invocation
    planned_decoded_daily_records: Literal[251] = 251
    planned_selected_endpoints: Literal[12] = 12

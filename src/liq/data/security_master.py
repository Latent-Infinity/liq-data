from datetime import UTC, date, datetime
from typing import Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict, StrictStr, field_validator


class SecurityMasterRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    ts_effective: AwareDatetime
    ts_record: AwareDatetime
    ts_created: AwareDatetime
    listing_id: StrictStr
    security_id: StrictStr
    issuer_id: StrictStr
    exchange: StrictStr

    @field_validator("listing_id", "security_id", "issuer_id", "exchange")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("reference identifier must be nonempty and unpadded")
        return value

    @field_validator("ts_effective", "ts_record", "ts_created", mode="before")
    @classmethod
    def validate_timestamp_type(cls, value: object) -> object:
        if not isinstance(value, (str, datetime)):
            raise ValueError("reference timestamp must be an aware datetime or ISO string")
        return value

    @field_validator("ts_effective", "ts_record", "ts_created")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


@runtime_checkable
class SecurityMasterProvider(Protocol):
    def get_security_master(
        self,
        symbols: tuple[str, ...],
        start: date,
        end: date,
        *,
        max_records: int,
    ) -> list[SecurityMasterRecord]: ...


def validate_security_master_request(
    symbols: tuple[str, ...], start: date, end: date, max_records: int
) -> None:
    if type(symbols) is not tuple or not 1 <= len(symbols) <= 2000:
        raise ValueError("security-master request requires a bounded symbol tuple")
    if any(
        not isinstance(symbol, str)
        or not symbol
        or symbol != symbol.strip()
        or symbol.upper() in {"ALL_SYMBOLS", "*"}
        or "," in symbol
        for symbol in symbols
    ):
        raise ValueError("security-master request requires explicit symbols")
    if len({symbol.upper() for symbol in symbols}) != len(symbols):
        raise ValueError("security-master request contains duplicate symbols")
    if type(start) is not date or type(end) is not date or start > end or end == date.max:
        raise ValueError("security-master request requires valid inclusive date bounds")
    if type(max_records) is not int or max_records < 1:
        raise ValueError("security-master record limit must be a positive integer")

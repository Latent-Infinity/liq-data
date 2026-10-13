"""Bounded dataset availability and definition-cost metadata; no records."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Final, Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from liq.data.providers._databento_transport import MetadataRefused, fetch_bounded

_ENDPOINT: Final = "https://hist.databento.com/v0/metadata.get_dataset_range"
_COST_ENDPOINT: Final = "https://hist.databento.com/v0/metadata.get_cost"
_REQUEST_SECONDS = 20.0
_MAX_RESPONSE_BYTES: Final = 65536
_UTC = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(?:Z|\+00:00)")


class MetadataDataset(StrEnum):
    MINI = "EQUS.MINI"
    SUMMARY = "EQUS.SUMMARY"
    NASDAQ = "XNAS.ITCH"


@dataclass(frozen=True, slots=True)
class MetadataRequest:
    dataset: MetadataDataset

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, MetadataDataset):
            raise MetadataRefused("dataset_scope")


@dataclass(frozen=True, slots=True)
class RawMetadataResponse:
    dataset: MetadataDataset
    http_status: int
    body: bytes = field(repr=False)
    content_type: str = field(repr=False)
    content_encoding: str = field(repr=False)


def _utc_key(value: str) -> str:
    match = _UTC.fullmatch(value)
    if match is None:
        raise ValueError("UTC ISO timestamp required")
    datetime.fromisoformat(match[1])
    return match[1] + "." + (match[2] or "").ljust(9, "0")


@dataclass(frozen=True, slots=True)
class DefinitionCostRequest:
    start: str
    end: str
    limit: int
    dataset: MetadataDataset = MetadataDataset.NASDAQ

    def __post_init__(self) -> None:
        try:
            if (
                self.dataset is not MetadataDataset.NASDAQ
                or type(self.limit) is not int
                or not 1 <= self.limit <= 1000
                or _utc_key(self.start)[10:] != "T00:00:00.000000000"
                or _utc_key(self.end)[10:] != "T00:00:00.000000000"
                or datetime.fromisoformat(self.end[:19]) - datetime.fromisoformat(self.start[:19])
                != timedelta(days=1)
            ):
                raise ValueError("unsupported definition cost scope")
        except (ValueError, TypeError) as exc:
            raise MetadataRefused("cost_request_scope") from exc


class DefinitionCostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    dataset: MetadataDataset
    start: str
    end: str
    record_limit: int
    estimated_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    complete_universe: Literal[False] = False


class AvailabilityRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        _utc_key(value)
        return value

    @model_validator(mode="after")
    def ordered_range(self) -> Self:
        if _utc_key(self.start) >= _utc_key(self.end):
            raise ValueError("empty or reversed exclusive range")
        return self


class SchemaAvailability(AvailabilityRange):
    schema_name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_-]+$")


class DatasetDescriptor(AvailabilityRange):
    dataset: MetadataDataset
    schemas: tuple[SchemaAvailability, ...]
    end_inclusive: Literal[False] = False


class _WireRange(AvailabilityRange):
    # Temporary JSON parsing model; the adopted descriptor contains only tuples.
    schemas: dict[str, AvailabilityRange] = Field(alias="schema", max_length=100)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate metadata key")
        result[key] = value
    return result


def _nonfinite(_: str) -> None:
    raise ValueError("nonfinite JSON refused")


def _check_response(response: RawMetadataResponse) -> None:
    if response.http_status != 200:
        raise MetadataRefused(
            "http_refused", http_status=response.http_status, received_bytes=len(response.body)
        )
    if response.content_encoding.lower() != "identity":
        raise MetadataRefused("unexpected_content_encoding")
    if response.content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise MetadataRefused("unexpected_content_type")
    if len(response.body) > _MAX_RESPONSE_BYTES:
        raise MetadataRefused("response_byte_limit")


def parse_dataset_range(response: RawMetadataResponse) -> DatasetDescriptor:
    """Parse retained metadata exactly; schema ranges never inherit dataset bounds."""
    _check_response(response)
    try:
        document = json.loads(
            response.body, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite
        )
        parsed = _WireRange.model_validate(document)
        schemas = tuple(
            SchemaAvailability(schema_name=name, start=value.start, end=value.end)
            for name, value in sorted(parsed.schemas.items())
        )
        if any(
            _utc_key(value.start) < _utc_key(parsed.start)
            or _utc_key(value.end) > _utc_key(parsed.end)
            for value in schemas
        ):
            raise MetadataRefused("schema_range_scope")
        return DatasetDescriptor(
            dataset=response.dataset, start=parsed.start, end=parsed.end, schemas=schemas
        )
    except (ValueError, ValidationError) as exc:
        if isinstance(exc, MetadataRefused):
            raise
        raise MetadataRefused("metadata_descriptor") from exc


def parse_definition_cost(
    response: RawMetadataResponse, request: DefinitionCostRequest
) -> DefinitionCostEstimate:
    """Preserve the exact numeric estimate; this never authorizes record access."""
    request.__post_init__()
    if response.dataset is not request.dataset:
        raise MetadataRefused("cost_response_scope")
    _check_response(response)
    try:
        amount = json.loads(
            response.body,
            parse_float=Decimal,
            parse_int=Decimal,
            parse_constant=_nonfinite,
        )
        if not isinstance(amount, Decimal) or not amount.is_finite() or amount < 0:
            raise ValueError("finite nonnegative numeric estimate required")
        return DefinitionCostEstimate(
            dataset=request.dataset,
            start=request.start,
            end=request.end,
            record_limit=request.limit,
            estimated_usd=amount,
        )
    except (ValueError, InvalidOperation) as exc:
        raise MetadataRefused("cost_descriptor") from exc


def fetch_dataset_range(
    request: MetadataRequest,
    api_key: str,
    *,
    on_http_attempt: Callable[[], None],
    transport: httpx.AsyncBaseTransport | None = None,
) -> RawMetadataResponse:
    """One GET without retry/redirect; caller retains raw bytes before parsing."""
    if not isinstance(request, MetadataRequest):
        raise MetadataRefused("dataset_scope")
    return _fetch_metadata(request, api_key, on_http_attempt, transport)


def fetch_definition_cost(
    request: DefinitionCostRequest,
    api_key: str,
    *,
    on_http_attempt: Callable[[], None],
    transport: httpx.AsyncBaseTransport | None = None,
) -> RawMetadataResponse:
    """One documented cost GET; never SDK POST, records or automatic fallback."""
    if not isinstance(request, DefinitionCostRequest):
        raise MetadataRefused("cost_request_scope")
    request.__post_init__()
    return _fetch_metadata(request, api_key, on_http_attempt, transport)


def _fetch_metadata(
    request: MetadataRequest | DefinitionCostRequest,
    api_key: str,
    on_http_attempt: Callable[[], None],
    transport: httpx.AsyncBaseTransport | None,
) -> RawMetadataResponse:
    if not api_key:
        raise MetadataRefused("missing_credentials")
    if not isinstance(request.dataset, MetadataDataset):
        raise MetadataRefused("dataset_scope")
    endpoint = _ENDPOINT
    params = {"dataset": request.dataset.value}
    if isinstance(request, DefinitionCostRequest):
        endpoint = _COST_ENDPOINT
        params.update(
            schema="definition",
            symbols="ALL_SYMBOLS",
            stype_in="raw_symbol",
            start=request.start,
            end=request.end,
            limit=str(request.limit),
        )
    raw = fetch_bounded(
        method="GET",
        endpoint=endpoint,
        parameters=params,
        api_key=api_key,
        on_http_attempt=on_http_attempt,
        max_response_bytes=_MAX_RESPONSE_BYTES,
        deadline_seconds=_REQUEST_SECONDS,
        transport=transport,
    )
    return RawMetadataResponse(
        request.dataset,
        raw.http_status,
        raw.body,
        raw.content_type,
        raw.content_encoding or "identity",
    )

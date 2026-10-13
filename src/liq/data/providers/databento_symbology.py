"""Bounded, guarded historical identity metadata; no issuer or universe admission."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

from liq.data.lockbox import FoldWindows, LockboxGuard, LockboxLedger
from liq.data.providers._databento_transport import BoundedResponse, MetadataRefused, fetch_bounded
from liq.data.providers.databento_metadata import _nonfinite, _unique_pairs

SYMBOLOGY_DATASET: Final = "databento_xnas_symbology_metadata"
SYMBOLOGY_ARM: Final = "classical_ranker_symbology_source_char"
SYMBOLOGY_LEDGER: Final = LockboxLedger(
    version="classical-ranker-symbology-characterization-2026-10-03",
    datasets=MappingProxyType(
        {
            SYMBOLOGY_DATASET: FoldWindows(
                characterization=(date(2018, 5, 1), date(2018, 5, 1)),
                lockbox_start=date(2026, 1, 1),
                allowed_purposes=frozenset({"characterization"}),
            ),
        }
    ),
)
_ENDPOINT: Final = "https://hist.databento.com/v0/symbology.resolve"
_MAX_RESPONSE_BYTES: Final = 4 * 1024 * 1024
_MAX_SYMBOLS: Final = 100000
_MAX_INTERVALS: Final = 100000
_REQUEST_SECONDS = 20.0


class SymbologyRefused(MetadataRefused):
    """Safe reason only; raw provider bodies never enter the exception."""


@dataclass(frozen=True, slots=True)
class SymbologyRequest:
    dataset: str = "XNAS.ITCH"
    symbols: str = "ALL_SYMBOLS"
    start_date: str = "2018-05-01"
    end_date: str = "2018-05-02"

    def parameters(self) -> dict[str, str]:
        if (
            self.dataset != "XNAS.ITCH"
            or self.symbols != "ALL_SYMBOLS"
            or self.start_date != "2018-05-01"
            or self.end_date != "2018-05-02"
        ):
            raise SymbologyRefused("request_scope")
        return {
            "dataset": self.dataset,
            "symbols": self.symbols,
            "stype_in": "raw_symbol",
            "stype_out": "instrument_id",
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


class ResolutionInterval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    d0: StrictStr
    d1: StrictStr
    s: StrictStr = Field(min_length=1, max_length=32, pattern=r"^[0-9]+$")


class _Wire(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    result: dict[StrictStr, tuple[ResolutionInterval, ...]] = Field(max_length=_MAX_SYMBOLS)
    symbols: tuple[StrictStr, ...] = Field(max_length=_MAX_SYMBOLS)
    stype_in: Literal["raw_symbol"]
    stype_out: Literal["instrument_id"]
    start_date: Literal["2018-05-01"]
    end_date: Literal["2018-05-02"]
    partial: tuple[StrictStr, ...] = Field(max_length=_MAX_SYMBOLS)
    not_found: tuple[StrictStr, ...] = Field(max_length=_MAX_SYMBOLS)
    message: StrictStr = Field(max_length=1024)
    status: StrictInt = Field(ge=0, le=0)


@dataclass(frozen=True, slots=True)
class SymbolMapping:
    raw_symbol: str
    intervals: tuple[ResolutionInterval, ...]


@dataclass(frozen=True, slots=True)
class SymbologySnapshot:
    mappings: tuple[SymbolMapping, ...] = field(repr=False)
    symbols: tuple[str, ...] = field(repr=False)
    partial: tuple[str, ...] = field(repr=False)
    not_found: tuple[str, ...] = field(repr=False)
    start_date: str
    end_date: str
    completeness_claim: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"
    pit_vintage_claim: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"


def parse_symbology(response: BoundedResponse, request: SymbologyRequest) -> SymbologySnapshot:
    """Interpret retained bytes; preserve every mapping and unresolved classification."""
    request.parameters()
    if response.http_status != 200:
        raise SymbologyRefused(
            "http_refused", http_status=response.http_status, received_bytes=len(response.body)
        )
    if response.content_encoding.lower() not in ("", "identity"):
        raise SymbologyRefused("unexpected_content_encoding")
    if response.content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise SymbologyRefused("unexpected_content_type")
    if len(response.body) > _MAX_RESPONSE_BYTES:
        raise SymbologyRefused("response_byte_limit")
    try:
        document = json.loads(
            response.body, object_pairs_hook=_unique_pairs, parse_constant=_nonfinite
        )
        parsed = _Wire.model_validate(document)
        if sum(len(intervals) for intervals in parsed.result.values()) > _MAX_INTERVALS:
            raise SymbologyRefused("interval_limit")
        for collection in (parsed.symbols, parsed.partial, parsed.not_found):
            if len(collection) != len(set(collection)):
                raise SymbologyRefused("classification_duplicate")
        if set(parsed.not_found) & (set(parsed.result) | set(parsed.partial)):
            raise SymbologyRefused("classification_conflict")
        if parsed.symbols != ("ALL_SYMBOLS",) and not (
            set(parsed.result) | set(parsed.partial) | set(parsed.not_found)
        ) <= set(parsed.symbols):
            raise SymbologyRefused("symbol_scope")
        if any(
            not symbol.strip() or len(symbol) > 128
            for symbol in (
                *parsed.result,
                *parsed.symbols,
                *parsed.partial,
                *parsed.not_found,
            )
        ):
            raise SymbologyRefused("symbol_descriptor")
        mappings = []
        for symbol, intervals in sorted(parsed.result.items()):
            if len(intervals) != len(set(intervals)):
                raise SymbologyRefused("interval_duplicate")
            for interval in intervals:
                # Do not clip provider intervals or invent a missing observation.
                if not request.start_date <= interval.d0 < interval.d1 <= request.end_date:
                    raise SymbologyRefused("interval_scope")
                date.fromisoformat(interval.d0)
                date.fromisoformat(interval.d1)
            ordered = sorted(intervals, key=lambda value: (value.d0, value.d1, value.s))
            if any(left.d1 > right.d0 for left, right in zip(ordered, ordered[1:], strict=False)):
                raise SymbologyRefused("interval_overlap")
            mappings.append(SymbolMapping(symbol, intervals))
        return SymbologySnapshot(
            tuple(mappings),
            parsed.symbols,
            parsed.partial,
            parsed.not_found,
            parsed.start_date,
            parsed.end_date,
        )
    except SymbologyRefused:
        raise
    except (ValueError, ValidationError):
        raise SymbologyRefused("symbology_descriptor") from None


def _load_api_key() -> str:
    from liq.data.settings import LiqDataSettings

    return LiqDataSettings().databento_api_key or ""


def fetch_symbology(
    request: SymbologyRequest,
    *,
    guard: LockboxGuard,
    purpose: str,
    arm_id: str,
    expected_api_key_sha256: str,
    on_http_attempt: Callable[[], None],
    final_portfolio_review: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BoundedResponse:
    """Exact free metadata POST; source-authorized worker retains raw before parsing."""
    if type(request) is not SymbologyRequest:
        raise SymbologyRefused("request_scope")
    parameters = request.parameters()
    if (
        type(guard) is not LockboxGuard
        or guard.ledger is not SYMBOLOGY_LEDGER
        or purpose != "characterization"
        or arm_id != SYMBOLOGY_ARM
        or final_portfolio_review
    ):
        raise SymbologyRefused("research_scope")
    guard.assert_period_allowed(
        SYMBOLOGY_DATASET,
        date(2018, 5, 1),
        date(2018, 5, 1),
        purpose=purpose,
        arm_id=arm_id,
        final_portfolio_review=False,
    )
    api_key = _load_api_key()
    if not api_key or hashlib.sha256(api_key.encode()).hexdigest() != expected_api_key_sha256:
        raise SymbologyRefused("credential_context")
    try:
        return fetch_bounded(
            method="POST",
            endpoint=_ENDPOINT,
            parameters=parameters,
            api_key=api_key,
            on_http_attempt=on_http_attempt,
            max_response_bytes=_MAX_RESPONSE_BYTES,
            deadline_seconds=_REQUEST_SECONDS,
            transport=transport,
        )
    except MetadataRefused as exc:
        raise SymbologyRefused(
            exc.reason, http_status=exc.http_status, received_bytes=exc.received_bytes
        ) from None

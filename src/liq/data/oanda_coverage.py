"""OANDA timestamp comparison and bounded request-scope diagnostics.

Capture calls the supplied guarded probe and intercepts its transport metadata.
Source access requires a separately reviewed scope; importing does not fetch data.
"""

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import cast
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import httpx
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import liq.data.service as data_service
from liq.data.service import BarCoverageObservation


class RequestBoundaryError(Exception):
    """An attempted metadata request or comparison left its reviewed scope."""


@dataclass(frozen=True, slots=True)
class RequestScope:
    """The one permitted UTC date and OANDA environment host."""

    host: str
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class CandleMetadata:
    """Only the time and completion bit from a raw OANDA candle."""

    time: datetime
    complete: bool


class _CandlePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    time: str = Field(strict=True)
    complete: bool = Field(strict=True)


class _ResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    candles: tuple[_CandlePayload, ...]


@dataclass(frozen=True, slots=True)
class CoverageDifference:
    """Disjoint explanations for absent adapter-accepted minute timestamps."""

    provider_absent: tuple[datetime, ...]
    raw_incomplete: tuple[datetime, ...]
    adapter_dropped: tuple[datetime, ...]


@dataclass(slots=True)
class RequestTelemetry:
    """Non-price provenance retained even when a guarded probe raises."""

    get_count: int = 0
    request_started_utc: str | None = None
    request_ended_utc: str | None = None
    request_from_utc: str | None = None
    request_to_utc: str | None = None
    http_status: int | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticCapture:
    """Only guarded coverage metadata, never an OHLCV frame or response body."""

    observation: BarCoverageObservation
    difference: CoverageDifference
    http_get_count: int


def capture_coverage(
    probe: Callable[[], BarCoverageObservation],
    scope: RequestScope,
    *,
    telemetry: RequestTelemetry | None = None,
) -> DiagnosticCapture:
    """Capture raw and adapter timestamp identities in one guarded probe call."""
    original_get = httpx.Client.get
    original_summary = data_service._summarize_bar_coverage
    deadline = time.monotonic() + 60.0
    raw: tuple[CandleMetadata, ...] | None = None
    accepted: tuple[datetime, ...] | None = None
    get_count = 0
    telemetry = telemetry if telemetry is not None else RequestTelemetry()

    def bounded_get(client: httpx.Client, url: str) -> httpx.Response:
        nonlocal raw, get_count
        require_in_scope_request(url, scope, request_count=get_count)
        request_started = time.monotonic()
        remaining = deadline - request_started
        if remaining <= 0:
            raise RequestBoundaryError("OANDA diagnostic deadline exhausted")
        request_budget = min(30.0, remaining)
        get_count += 1
        telemetry.get_count = get_count
        telemetry.request_from_utc = scope.start.astimezone(UTC).isoformat()
        telemetry.request_to_utc = scope.end.astimezone(UTC).isoformat()
        telemetry.request_started_utc = datetime.now(UTC).isoformat()
        try:
            response = original_get(client, url, timeout=request_budget, follow_redirects=False)
            telemetry.http_status = response.status_code
        finally:
            telemetry.request_ended_utc = datetime.now(UTC).isoformat()
        request_finished = time.monotonic()
        if request_finished - request_started > request_budget:
            raise RequestBoundaryError("OANDA request budget exceeded")
        if request_finished > deadline:
            raise RequestBoundaryError("OANDA diagnostic deadline exhausted")
        raw = extract_candle_metadata(response)
        return response

    def capture_summary(
        frame: pl.DataFrame,
        *,
        provider: str,
        symbol: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> BarCoverageObservation:
        nonlocal accepted
        observation = original_summary(
            frame,
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        accepted = (
            cast(tuple[datetime, ...], tuple(frame["timestamp"].to_list()))
            if not frame.is_empty()
            else ()
        )
        canonical = "\n".join(item.astimezone(UTC).isoformat() for item in sorted(accepted))
        if hashlib.sha256(canonical.encode()).hexdigest() != observation.timestamp_sha256:
            raise RequestBoundaryError("adapter timestamp summary identity mismatch")
        return observation

    with (
        patch.object(httpx.Client, "get", bounded_get),
        patch.object(data_service, "_summarize_bar_coverage", capture_summary),
    ):
        observation = probe()

    if raw is None or accepted is None or get_count != 1:
        raise RequestBoundaryError("diagnostic did not capture one complete GET and summary")
    difference = compare_timestamps(raw, accepted, scope.start, scope.end)
    if time.monotonic() > deadline:
        raise RequestBoundaryError("OANDA diagnostic deadline exhausted")
    return DiagnosticCapture(
        observation=observation,
        difference=difference,
        http_get_count=get_count,
    )


def extract_candle_metadata(response: httpx.Response) -> tuple[CandleMetadata, ...]:
    """Project a successful response to timestamps and completion flags only."""
    if response.status_code != httpx.codes.OK:
        raise RequestBoundaryError("OANDA request was not successful")
    try:
        payload = _ResponsePayload.model_validate_json(response.content)
    except ValidationError as exc:
        raise RequestBoundaryError("invalid OANDA candle metadata") from exc
    metadata: list[CandleMetadata] = []
    for candle in payload.candles:
        if not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)",
            candle.time,
        ):
            raise RequestBoundaryError("OANDA candle timestamp is not a UTC minute")
        try:
            parsed = datetime.fromisoformat(candle.time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RequestBoundaryError("invalid OANDA candle timestamp") from exc
        metadata.append(CandleMetadata(time=parsed.astimezone(UTC), complete=candle.complete))
    return tuple(metadata)


def require_in_scope_request(url: str, scope: RequestScope, *, request_count: int) -> None:
    """Refuse any GET outside the exact one-day midpoint M1 request."""
    parsed = urlsplit(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    if (
        request_count != 0
        or parsed.scheme != "https"
        or parsed.netloc != scope.host
        or parsed.path != "/v3/instruments/EUR_USD/candles"
        or parsed.fragment
        or set(query) != {"from", "to", "granularity", "price"}
        or query["granularity"] != ["M1"]
        or query["price"] != ["M"]
        or len(query["from"]) != 1
        or len(query["to"]) != 1
    ):
        raise RequestBoundaryError("unapproved OANDA request")
    try:
        start = datetime.fromisoformat(query["from"][0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(query["to"][0].replace("Z", "+00:00"))
    except ValueError as exc:
        raise RequestBoundaryError("invalid UTC request bounds") from exc
    if start != scope.start or end != scope.end:
        raise RequestBoundaryError("unapproved UTC request bounds")


def compare_timestamps(
    raw: tuple[CandleMetadata, ...],
    accepted: tuple[datetime, ...],
    start: datetime,
    end: datetime,
) -> CoverageDifference:
    """Separate provider omissions, incomplete candles and adapter losses."""
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise RequestBoundaryError("invalid UTC comparison window")
    if any(item.time.tzinfo is None for item in raw) or any(
        item.tzinfo is None for item in accepted
    ):
        raise RequestBoundaryError("raw or accepted timestamp lacks timezone")
    grid: set[datetime] = set()
    minute = start.astimezone(UTC)
    while minute <= end.astimezone(UTC):
        grid.add(minute)
        minute += timedelta(minutes=1)
    raw_times = {item.time.astimezone(UTC) for item in raw}
    complete = {item.time.astimezone(UTC) for item in raw if item.complete}
    incomplete = {item.time.astimezone(UTC) for item in raw if not item.complete}
    accepted_times = {item.astimezone(UTC) for item in accepted}
    if (
        complete & incomplete
        or len(raw_times) != len(raw)
        or len(accepted_times) != len(accepted)
        or not raw_times <= grid
        or not accepted_times <= complete
    ):
        raise RequestBoundaryError("raw and accepted timestamp identities disagree")
    return CoverageDifference(
        provider_absent=tuple(sorted(grid - raw_times)),
        raw_incomplete=tuple(sorted(incomplete)),
        adapter_dropped=tuple(sorted(complete - accepted_times)),
    )

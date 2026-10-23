import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictInt, StrictStr

Reason = Literal[
    "BODY_BYTE_BUDGET",
    "INVALID_CUTOFF",
    "DUPLICATE_JSON_KEY",
    "NONFINITE_JSON",
    "METADATA_SCHEMA",
    "INVALID_REVISION_CLOCK",
    "REVISION_AFTER_CUTOFF",
    "UNSUPPORTED_SHA1_REPRESENTATION",
]


class WikipediaRevisionRefused(ValueError):
    def __init__(self, reason: Reason) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class WikipediaRevisionLocator:
    page_id: int
    namespace: int
    title: str
    revision_id: int
    parent_revision_id: int
    revision_timestamp_literal: str
    revision_timestamp_utc: datetime
    declared_content_bytes: int
    provider_declared_sha1: str
    continuation_present: bool
    content_verified: Literal[False] = False


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class _Revision(_StrictModel):
    revid: Annotated[StrictInt, Field(ge=1)]
    parentid: Annotated[StrictInt, Field(ge=0)]
    timestamp: StrictStr
    size: Annotated[StrictInt, Field(ge=0)]
    sha1: StrictStr


class _Page(_StrictModel):
    pageid: Annotated[StrictInt, Field(ge=1)]
    ns: Annotated[StrictInt, Field(ge=0, le=0)]
    title: Literal["List of S&P 500 companies"]
    revisions: Annotated[list[_Revision], Field(min_length=1, max_length=1)]


class _Query(_StrictModel):
    pages: Annotated[list[_Page], Field(min_length=1, max_length=1)]


_Token = Annotated[StrictStr, Field(min_length=1, max_length=512, pattern=r"^[\x20-\x7e]+$")]


class _Continuation(_StrictModel):
    rvcontinue: _Token
    continuation: _Token = Field(alias="continue")


class _Response(_StrictModel):
    batchcomplete: StrictBool = False
    query: _Query
    continuation: _Continuation | None = Field(default=None, alias="continue")


def _unique_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise WikipediaRevisionRefused("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _nonfinite(_value: str) -> JsonValue:
    raise WikipediaRevisionRefused("NONFINITE_JSON")


def parse_revision_metadata(raw: bytes, *, cutoff: datetime) -> WikipediaRevisionLocator:
    """Project metadata only; callers must retain original bytes before invoking this parser."""
    if cutoff.tzinfo is None or cutoff.utcoffset() != timedelta(0) or cutoff.microsecond:
        raise WikipediaRevisionRefused("INVALID_CUTOFF")
    if not raw or len(raw) >= 64 * 1024:
        raise WikipediaRevisionRefused("BODY_BYTE_BUDGET")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_nonfinite
        )
        response = _Response.model_validate(value)
        if "continue" in value and value["continue"] is None:
            raise WikipediaRevisionRefused("METADATA_SCHEMA")
    except WikipediaRevisionRefused:
        raise
    except (ValueError, RecursionError):
        raise WikipediaRevisionRefused("METADATA_SCHEMA") from None
    page = response.query.pages[0]
    revision = page.revisions[0]
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision.sha1) is None:
        raise WikipediaRevisionRefused("UNSUPPORTED_SHA1_REPRESENTATION")
    if (
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", revision.timestamp)
        is None
    ):
        raise WikipediaRevisionRefused("INVALID_REVISION_CLOCK")
    try:
        timestamp = datetime.strptime(revision.timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        raise WikipediaRevisionRefused("INVALID_REVISION_CLOCK") from None
    if timestamp > cutoff:
        raise WikipediaRevisionRefused("REVISION_AFTER_CUTOFF")
    return WikipediaRevisionLocator(
        page_id=page.pageid,
        namespace=page.ns,
        title=page.title,
        revision_id=revision.revid,
        parent_revision_id=revision.parentid,
        revision_timestamp_literal=revision.timestamp,
        revision_timestamp_utc=timestamp,
        declared_content_bytes=revision.size,
        provider_declared_sha1=revision.sha1,
        continuation_present=response.continuation is not None,
    )

"""Bounded identity entity bytes; completion alone never authorizes retention."""

import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Final, Literal

import httpx

MAX_SOURCE_BODY_BYTES: Final = 2_097_152
SOURCE_CHUNK_BYTES: Final = 65_536
BodyFailureReason = Literal[
    "HTTP_ERROR",
    "ENCODING_REFUSED",
    "LENGTH_REFUSED",
    "BODY_LIMIT",
    "LENGTH_MISMATCH",
    "STREAM_REFUSED",
    "TRANSPORT_ERROR",
]


@dataclass(frozen=True, slots=True)
class SourceBody:
    """Complete bytes awaiting envelope, secret, identity and retention checks."""

    body: bytes = field(repr=False)
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceBodyFailure:
    """Safe transport-only evidence, with no body, headers, URL or error text."""

    reason: BodyFailureReason
    http_status: int
    received_bytes: int


def _declared_length(response: httpx.Response) -> int | None | BodyFailureReason:
    lengths = response.headers.get_list("content-length")
    if not lengths:
        return None
    if len(lengths) != 1 or re.fullmatch(r"[0-9]+", lengths[0]) is None:
        return "LENGTH_REFUSED"
    normalized = lengths[0].lstrip("0") or "0"
    maximum = str(MAX_SOURCE_BODY_BYTES)
    if len(normalized) > len(maximum) or (len(normalized) == len(maximum) and normalized > maximum):
        return "BODY_LIMIT"
    return int(normalized)


def _read_open_body(response: httpx.Response) -> SourceBody | SourceBodyFailure:
    status = response.status_code
    if response.is_stream_consumed or response.is_closed:
        return SourceBodyFailure("STREAM_REFUSED", status, response.num_bytes_downloaded)
    if status != 200:
        return SourceBodyFailure("HTTP_ERROR", status, 0)
    encodings = response.headers.get_list("content-encoding")
    if encodings and encodings != ["identity"]:
        return SourceBodyFailure("ENCODING_REFUSED", status, 0)
    declared = _declared_length(response)
    if isinstance(declared, str):
        return SourceBodyFailure(declared, status, 0)
    chunks = bytearray()
    try:
        for chunk in response.iter_raw(chunk_size=SOURCE_CHUNK_BYTES):
            if len(chunks) + len(chunk) > MAX_SOURCE_BODY_BYTES:
                return SourceBodyFailure("BODY_LIMIT", status, response.num_bytes_downloaded)
            chunks.extend(chunk)
    except httpx.RequestError:
        return SourceBodyFailure("TRANSPORT_ERROR", status, response.num_bytes_downloaded)
    except httpx.StreamError:
        return SourceBodyFailure("STREAM_REFUSED", status, response.num_bytes_downloaded)
    if declared is not None and len(chunks) != declared:
        return SourceBodyFailure("LENGTH_MISMATCH", status, response.num_bytes_downloaded)
    body = bytes(chunks)
    return SourceBody(body, sha256(body).hexdigest())


def read_source_body(response: httpx.Response) -> SourceBody | SourceBodyFailure:
    """Read at most 2 MiB in 64 KiB raw chunks, always closing the response.

    No GET is made here. The collector must supervise the whole request with
    its separately verified hard deadline. Download counts include chunks
    observed by HTTPX even when its chunker has not yet yielded them.
    """
    close_failed = False
    try:
        result = _read_open_body(response)
    finally:
        try:
            response.close()
        except (httpx.RequestError, httpx.StreamError):
            close_failed = True
    if close_failed:
        return SourceBodyFailure(
            "TRANSPORT_ERROR", response.status_code, response.num_bytes_downloaded
        )
    return result

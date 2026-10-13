"""Private bounded HTTP mechanics; public owning wrappers authorize each route."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, assert_never

import httpx


class MetadataRefused(ValueError):
    """Safe reason only; response bodies and authentication never enter errors."""

    def __init__(
        self, reason: str, *, http_status: int | None = None, received_bytes: int = 0
    ) -> None:
        self.reason = reason
        self.http_status = http_status
        self.received_bytes = received_bytes
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class BoundedResponse:
    http_status: int
    body: bytes = field(repr=False)
    content_type: str = field(repr=False)
    content_encoding: str = field(repr=False)


class _AttemptTransport(httpx.AsyncBaseTransport):
    def __init__(self, delegate: httpx.AsyncBaseTransport, witness: Callable[[], None]) -> None:
        self._delegate = delegate
        self._witness = witness
        self._entered = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._entered:
            raise MetadataRefused("repeated_transport_entry")
        self._entered = True
        self._witness()
        return await self._delegate.handle_async_request(request)

    async def aclose(self) -> None:
        await self._delegate.aclose()


def fetch_bounded(
    *,
    method: Literal["GET", "POST"],
    endpoint: str,
    parameters: dict[str, str],
    api_key: str,
    on_http_attempt: Callable[[], None],
    max_response_bytes: int,
    deadline_seconds: float,
    transport: httpx.AsyncBaseTransport | None,
) -> BoundedResponse:
    """One stream, no retries/redirects, bounded before and during receipt."""
    if not api_key:
        raise MetadataRefused("missing_credentials")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise MetadataRefused("requires_sync_context")

    async def read() -> BoundedResponse:
        delegate = transport or httpx.AsyncHTTPTransport(retries=0)
        witnessed = _AttemptTransport(delegate, on_http_attempt)
        match method:
            case "GET":
                query, form = parameters, None
            case "POST":
                query, form = None, parameters
            case unreachable:
                assert_never(unreachable)
        async with (
            httpx.AsyncClient(
                transport=witnessed,
                timeout=httpx.Timeout(deadline_seconds),
                auth=httpx.BasicAuth(api_key, ""),
                follow_redirects=False,
                trust_env=False,
            ) as client,
            client.stream(
                method,
                endpoint,
                params=query,
                data=form,
                headers={"Accept": "application/json", "Accept-Encoding": "identity"},
                follow_redirects=False,
            ) as response,
        ):
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise MetadataRefused(
                    "unexpected_content_encoding", http_status=response.status_code
                )
            length = response.headers.get("content-length")
            if length is not None and (
                not length.isascii()
                or not length.isdecimal()
                or len(length) > 10
                or int(length) > max_response_bytes
            ):
                raise MetadataRefused("response_byte_limit", http_status=response.status_code)
            body = bytearray()
            async for chunk in response.aiter_raw(chunk_size=1024):
                if len(body) + len(chunk) > max_response_bytes:
                    raise MetadataRefused(
                        "response_byte_limit",
                        http_status=response.status_code,
                        received_bytes=len(body) + len(chunk),
                    )
                body.extend(chunk)
            if length is not None and len(body) != int(length):
                raise MetadataRefused(
                    "content_length_mismatch",
                    http_status=response.status_code,
                    received_bytes=len(body),
                )
            return BoundedResponse(
                response.status_code,
                bytes(body),
                response.headers.get("content-type", ""),
                response.headers.get("content-encoding", ""),
            )

    try:
        return asyncio.run(asyncio.wait_for(read(), timeout=deadline_seconds))
    except TimeoutError as exc:
        raise MetadataRefused("request_deadline") from exc
    except httpx.HTTPError as exc:
        raise MetadataRefused("transport_error") from exc

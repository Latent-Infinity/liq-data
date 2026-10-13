"""Fail-closed HTTP isolation for nonfinancial announcement controls."""

import httpx
import pytest


async def _refuse_async(
    _transport: httpx.AsyncHTTPTransport, _request: httpx.Request
) -> httpx.Response:
    raise AssertionError("announcement controls must never use real HTTP transport")


def _refuse_sync(_transport: httpx.HTTPTransport, _request: httpx.Request) -> httpx.Response:
    raise AssertionError("announcement controls must never use real HTTP transport")


@pytest.fixture(autouse=True)
def isolated_announcement_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _refuse_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _refuse_sync)

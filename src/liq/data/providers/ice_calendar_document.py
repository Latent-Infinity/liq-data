"""Fixed ICE calendar document bytes; no calendar or historical admission."""

import httpx

ICE_URL = "https://ir.theice.com/press/news-details/2022/NYSE-Group-Announces-2023-2024-and-2025-Holiday-and-Early-Closings-Calendar/default.aspx"


def _capture(client: httpx.Client) -> bytes:
    limit = 4194304
    with client.stream(
        "GET",
        ICE_URL,
        headers={"Accept-Encoding": "identity"},
        timeout=20.0,
        follow_redirects=False,
    ) as response:
        if response.status_code != 200:
            raise ValueError("fixed_document_http_refused")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise ValueError("fixed_document_encoding_refused")
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                raise ValueError("fixed_document_length_refused") from None
            if length < 0 or length >= limit:
                raise ValueError("fixed_document_byte_budget")
        raw = bytearray()
        for chunk in response.iter_bytes(chunk_size=1024):
            raw.extend(chunk)
            if len(raw) >= limit:
                raise ValueError("fixed_document_byte_budget")
        return bytes(raw)


def capture_ice_calendar(client: httpx.Client | None = None) -> bytes:
    """One fixed GET, no redirects/retries/links, bounded identity-encoded bytes."""
    if client is not None:
        return _capture(client)
    with httpx.Client(follow_redirects=False) as owned:
        return _capture(owned)

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping

import requests
import zstandard
from databento.common.http import check_http_error
from databento.reference.api.security import SecurityMasterHttpAPI
from requests.auth import HTTPBasicAuth


class CaptureLimit(ValueError):
    pass


class BoundedSecurityMaster(SecurityMasterHttpAPI):
    def __init__(
        self,
        key: str,
        *,
        max_wire_bytes: int,
        max_decoded_bytes: int,
        max_records: int,
    ) -> None:
        super().__init__(key, "https://hist.databento.com")
        if any(
            type(value) is not int or value < 1
            for value in (max_wire_bytes, max_decoded_bytes, max_records)
        ):
            raise CaptureLimit("invalid_limits")
        self.max_wire_bytes = max_wire_bytes
        self.max_decoded_bytes = max_decoded_bytes
        self.max_records = max_records
        self.attempted = False
        self.raw_capture: bytes | None = None
        self.decoded_bytes = 0

    def _post(
        self,
        url: str,
        data: Mapping[str, object | None] | None = None,
        params: Iterable[tuple[str, str | None]] | None = None,
        basic_auth: bool = False,
    ) -> requests.Response:
        if self.attempted or url != self._base_url + ".get_range" or not basic_auth:
            raise CaptureLimit("unexpected_or_repeated_request")
        self.attempted = True
        self._check_api_key()
        deadline = time.monotonic() + 110
        with requests.post(
            url=url,
            data=data,
            params=params,
            headers=self._headers,
            auth=HTTPBasicAuth(self._key, ""),
            timeout=(10, 10),
            stream=True,
            allow_redirects=False,
        ) as response:
            if 300 <= response.status_code < 400:
                raise CaptureLimit("redirect_refused")
            if response.status_code != 200:
                response._content = b""
                check_http_error(response)
                raise CaptureLimit("unexpected_http_status")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise CaptureLimit("unexpected_content_encoding")
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > self.max_wire_bytes or time.monotonic() > deadline:
                    raise CaptureLimit("wire_or_time_limit")
                chunks.append(chunk)
            compressed = b"".join(chunks)
            try:
                content_size = zstandard.frame_content_size(compressed)
                if (
                    0 <= content_size < zstandard.CONTENTSIZE_UNKNOWN
                    and content_size > self.max_decoded_bytes
                ):
                    raise CaptureLimit("decoded_byte_limit")
                decoded = zstandard.ZstdDecompressor().decompress(
                    compressed,
                    max_output_size=self.max_decoded_bytes,
                    allow_extra_data=False,
                )
            except zstandard.ZstdError as exc:
                raise CaptureLimit("invalid_or_oversize_compression") from exc
            if len(decoded.splitlines()) > self.max_records:
                raise CaptureLimit("record_limit")
            if time.monotonic() > deadline:
                raise CaptureLimit("time_limit")
            self.decoded_bytes = len(decoded)
            self.raw_capture = compressed
            response._content = compressed
            response._content_consumed = True
            return response

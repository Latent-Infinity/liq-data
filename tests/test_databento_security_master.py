import io
from unittest.mock import patch

import pytest
import requests
import zstandard

from liq.data.providers.databento_security_master import (
    BoundedSecurityMaster,
    CaptureLimit,
)


def response(payload, status=200):
    result = requests.Response()
    result.status_code = status
    result.raw = io.BytesIO(payload)
    return result


def endpoint(**overrides):
    limits = {"max_wire_bytes": 1024, "max_decoded_bytes": 1024, "max_records": 2}
    limits.update(overrides)
    return BoundedSecurityMaster("test-key", **limits)


def test_bounded_transport_retains_exact_compressed_bytes():
    payload = zstandard.ZstdCompressor().compress(b'{"listing_id":"TEST"}\n')
    api = endpoint()
    with patch("requests.post", return_value=response(payload)) as post:
        result = api._post(api._base_url + ".get_range", data={}, basic_auth=True)
    assert result.content == payload == api.raw_capture
    assert api.decoded_bytes == len(b'{"listing_id":"TEST"}\n')
    assert post.call_args.kwargs["stream"] is True
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    "limits,payload",
    [
        ({"max_wire_bytes": 1}, b"too large"),
        ({"max_decoded_bytes": 2}, zstandard.ZstdCompressor().compress(b"{}\n")),
        ({"max_records": 1}, zstandard.ZstdCompressor().compress(b"{}\n{}\n")),
    ],
)
def test_limit_refuses_instead_of_returning_prefix(limits, payload):
    api = endpoint(**limits)
    with patch("requests.post", return_value=response(payload)), pytest.raises(CaptureLimit):
        api._post(api._base_url + ".get_range", data={}, basic_auth=True)
    assert api.raw_capture is None


def test_redirect_and_second_request_are_refused():
    api = endpoint()
    with patch("requests.post", return_value=response(b"", 302)) as post:
        with pytest.raises(CaptureLimit):
            api._post(api._base_url + ".get_range", data={}, basic_auth=True)
        with pytest.raises(CaptureLimit):
            api._post(api._base_url + ".get_range", data={}, basic_auth=True)
    assert post.call_count == 1


def test_unexpected_endpoint_is_refused_before_network():
    api = endpoint()
    with patch("requests.post") as post, pytest.raises(CaptureLimit):
        api._post("https://example.com", data={}, basic_auth=True)
    post.assert_not_called()


@pytest.mark.parametrize("payload", [b"invalid", zstandard.ZstdCompressor().compress(b"{}\n")[:-1]])
def test_malformed_or_truncated_compression_is_refused(payload):
    api = endpoint()
    with patch("requests.post", return_value=response(payload)), pytest.raises(CaptureLimit):
        api._post(api._base_url + ".get_range", data={}, basic_auth=True)
    assert api.raw_capture is None


def test_authorization_error_is_one_request_without_body_exposure():
    api = endpoint()
    from databento.common.error import BentoClientError

    with (
        patch("requests.post", return_value=response(b"private error text", 401)) as post,
        pytest.raises(BentoClientError) as error,
    ):
        api._post(api._base_url + ".get_range", data={}, basic_auth=True)
    assert error.value.http_status == 401
    assert "private error text" not in str(error.value)
    assert post.call_count == 1


def test_invalid_limits_and_http_encoding_are_refused():
    with pytest.raises(CaptureLimit):
        endpoint(max_records=0)
    result = response(b"{}")
    result.headers["Content-Encoding"] = "gzip"
    api = endpoint()
    with patch("requests.post", return_value=result), pytest.raises(CaptureLimit):
        api._post(api._base_url + ".get_range", data={}, basic_auth=True)

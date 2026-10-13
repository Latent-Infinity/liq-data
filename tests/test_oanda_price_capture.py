"""Intercepted empty envelopes verify transport, never native-price admission."""

import logging
from datetime import date

import httpx
import pytest

from liq.data.oanda_instrument_metadata import MATRIX_SYMBOLS
from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.providers.oanda import OandaProvider
from liq.data.providers.oanda_price_capture import PriceCaptureFailure, capture_source_prices
from liq.data.providers.oanda_source_body import SourceBody
from liq.data.providers.oanda_source_transport import MATRIX_DAYS


def provider() -> OandaProvider:
    return OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")


@pytest.mark.parametrize(
    ("symbol", "day"),
    [("GBP_JPY", date(2024, 1, 3)), ("EUR_USD", date(2025, 1, 3)), ("EUR_USD", date(2026, 1, 3))],
)
def test_direct_provider_refuses_forged_window_before_client(
    symbol: str, day: date, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a window constructed without the approved-case validator.
    source = provider()
    window = SourcePriceWindow.model_construct(symbol=symbol, day=day)

    def unexpected_client() -> httpx.Client:
        raise AssertionError("Refused windows must not resolve a transport client")

    monkeypatch.setattr(source, "_get_client", unexpected_client)
    # When the public provider capability receives the forged window.
    result = source.capture_source_prices(window)
    # Then the receipt is category-only, with no client or GET.
    assert result == PriceCaptureFailure("WINDOW_REFUSED", None, 0)
    assert source._client is None and symbol not in repr(result)


@pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
@pytest.mark.parametrize("day", MATRIX_DAYS)
def test_exact_account_get_streams_empty_envelope(symbol: str, day: date) -> None:
    # Given an approved window, authenticated real client and intercepted wire.
    source = provider()
    window = SourcePriceWindow(symbol=symbol, day=day)
    calls: list[httpx.Request] = []
    raw = f'{{"instrument":"{symbol}","granularity":"M1","candles":[]}}\n'.encode()

    def intercept(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        assert request.url.host == "api-fxtrade.oanda.com"
        assert request.url.path == f"/v3/accounts/dummy-account/instruments/{symbol}/candles"
        assert dict(request.url.params) == window.query()
        assert request.headers["Authorization"] == "Bearer dummy-token"
        assert request.headers["Accept-Encoding"] == "identity"
        assert set(request.extensions["timeout"].values()) == {30.0}
        return httpx.Response(200, stream=httpx.ByteStream(raw))

    with httpx.Client(
        transport=httpx.MockTransport(intercept),
        headers={"Authorization": "Bearer dummy-token"},
    ) as client:
        source._client = client
        # When the narrow provider transport completes.
        result = capture_source_prices(source, window)
    # Then only exact complete structural bytes are returned, not price acceptance.
    assert isinstance(result, SourceBody) and result.body == raw
    assert len(calls) == 1 and "dummy" not in repr(result)


@pytest.mark.parametrize("environment", ["practice", "typo"])
def test_non_live_environment_refuses_before_client_creation(environment: str) -> None:
    source = OandaProvider(
        api_key="dummy-token", account_id="dummy-account", environment=environment
    )
    result = capture_source_prices(
        source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
    )
    assert result == PriceCaptureFailure("ENDPOINT_REFUSED", None, 0)
    assert source._client is None


@pytest.mark.parametrize("target", ["https://example.invalid", "http://api-fxtrade.oanda.com"])
def test_alternate_base_url_refuses_before_client_creation(target: str) -> None:
    source = provider()
    source._base_url = target
    result = capture_source_prices(
        source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
    )
    assert result == PriceCaptureFailure("ENDPOINT_REFUSED", None, 0)
    assert source._client is None


@pytest.mark.parametrize("account", ["", "../another", "id?token=secret", "id/extra"])
def test_invalid_account_path_refuses_before_client_creation(account: str) -> None:
    source = OandaProvider(api_key="dummy-token", account_id=account, environment="live")
    result = capture_source_prices(
        source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
    )
    assert result == PriceCaptureFailure("ENDPOINT_REFUSED", None, 0)
    assert source._client is None


def test_empty_token_refuses_before_client_creation() -> None:
    source = OandaProvider(api_key="", account_id="dummy-account", environment="live")
    result = capture_source_prices(
        source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
    )
    assert result == PriceCaptureFailure("INVALID_SECRETS", None, 0)
    assert source._client is None


def test_non_ascii_token_cannot_leak_header_encoding_exception() -> None:
    source = OandaProvider(
        api_key="dummy-\u2603-token", account_id="dummy-account", environment="live"
    )
    result = capture_source_prices(
        source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
    )
    assert result == PriceCaptureFailure("INVALID_SECRETS", None, 0)
    assert source._client is None and "dummy" not in repr(result)


@pytest.mark.parametrize("status", [301, 401, 429, 500])
def test_http_failure_has_no_body_and_no_redirect_or_retry(status: int) -> None:
    source = provider()
    calls: list[httpx.Request] = []

    def intercept(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://example.invalid"},
            stream=httpx.ByteStream(b"private"),
        )

    with httpx.Client(transport=httpx.MockTransport(intercept), follow_redirects=True) as client:
        source._client = client
        result = capture_source_prices(
            source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
        )
    assert result == PriceCaptureFailure("HTTP_ERROR", status, 0)
    assert len(calls) == 1 and "private" not in repr(result)


def test_connect_error_has_no_exception_text_and_no_retry() -> None:
    source = provider()
    calls: list[httpx.Request] = []

    def intercept(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("dummy-token dummy-account", request=request)

    with httpx.Client(transport=httpx.MockTransport(intercept)) as client:
        source._client = client
        result = capture_source_prices(
            source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
        )
    assert result == PriceCaptureFailure("TRANSPORT_ERROR", None, 0)
    assert len(calls) == 1 and "dummy" not in repr(result)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"instrument":"GBP_USD","granularity":"M1","candles":[]}',
        b'{"instrument":"EUR_USD","granularity":"M1","candles":[],"extra":"dummy-token"}',
        b'{"instrument":"EUR_USD","granularity":"M1","candles":[],"extra":"dummy\\u002dtoken"}',
    ],
)
def test_non_publishable_envelope_never_returns_raw_body(raw: bytes) -> None:
    source = provider()

    def intercept(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(raw))

    with httpx.Client(transport=httpx.MockTransport(intercept)) as client:
        source._client = client
        result = capture_source_prices(
            source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3))
        )
    assert isinstance(result, PriceCaptureFailure)
    assert result.http_status == 200 and result.received_bytes == len(raw)
    assert not hasattr(result, "body") and "dummy" not in repr(result)


def test_httpx_account_path_is_redacted_and_filter_is_removed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = provider()
    logger = logging.getLogger("httpx")
    filters = tuple(logger.filters)

    def intercept(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    with (
        httpx.Client(transport=httpx.MockTransport(intercept)) as client,
        caplog.at_level(logging.INFO, logger="httpx"),
    ):
        source._client = client
        capture_source_prices(source, SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3)))
    assert "dummy-account" not in caplog.text and "dummy-token" not in caplog.text
    assert "<redacted>" in caplog.text and tuple(logger.filters) == filters

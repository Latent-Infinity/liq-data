"""Intercepted cohort transport uses empty envelopes, never financial fixtures."""

from datetime import date

import httpx
import pytest

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_instrument_metadata import MATRIX_SYMBOLS
from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.providers.oanda import OandaProvider
from liq.data.providers.oanda_price_capture import PriceCaptureFailure
from liq.data.providers.oanda_source_body import SourceBody


def cohort_window(symbol: str = "EUR_USD", day: date = date(2023, 1, 2)) -> CohortSourceWindow:
    return CohortSourceWindow(scope="london-fx-fix-development-source/v1", symbol=symbol, day=day)


@pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
@pytest.mark.parametrize("day", [date(2023, 1, 2), date(2023, 7, 3), date(2024, 3, 28)])
def test_cohort_get_preserves_fixed_account_query(symbol: str, day: date) -> None:
    # Given a real retained provider/client intercepted at the wire.
    source = OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")
    window = cohort_window(symbol, day)
    raw = f'{{"instrument":"{symbol}","granularity":"M1","candles":[]}}'.encode()
    calls: list[httpx.Request] = []

    def intercept(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET"
        assert request.url.path == f"/v3/accounts/dummy-account/instruments/{symbol}/candles"
        assert request.url.host == "api-fxtrade.oanda.com"
        assert dict(request.url.params) == window.query()
        assert request.headers["Accept-Encoding"] == "identity"
        assert request.headers["Authorization"] == "Bearer dummy-token"
        assert set(request.extensions["timeout"].values()) == {30.0}
        return httpx.Response(200, stream=httpx.ByteStream(raw))

    with httpx.Client(
        transport=httpx.MockTransport(intercept),
        headers={"Authorization": "Bearer dummy-token"},
    ) as client:
        source._client = client
        # When the distinct cohort capability receives a fixed case.
        result = source.capture_cohort_source_prices(window)
    # Then exact structural bytes are returned once, without admission.
    assert isinstance(result, SourceBody) and result.body == raw
    assert len(calls) == 1 and "dummy" not in repr(result)


@pytest.mark.parametrize("day", [date(2022, 1, 3), date(2023, 1, 7), date(2025, 1, 2)])
def test_cohort_forged_day_refuses_before_client(
    day: date, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given an instance copied without model validation.
    source = OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")
    window = cohort_window().model_copy(update={"day": day})

    def unexpected_client() -> httpx.Client:
        pytest.fail("CLIENT_RESOLUTION_BEFORE_SCOPE_REFUSAL")

    monkeypatch.setattr(source, "_get_client", unexpected_client)
    # When the direct provider capability reparses the closed cohort.
    result = source.capture_cohort_source_prices(window)
    # Then it refuses safely before client resolution.
    assert result == PriceCaptureFailure("WINDOW_REFUSED", None, 0)


@pytest.mark.parametrize("status", [301, 401, 429, 500])
def test_cohort_http_failure_has_no_retry_redirect_or_body(status: int) -> None:
    # Given a wire response whose body and redirect must not be followed.
    source = OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")
    calls: list[httpx.Request] = []

    def intercept(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://example.invalid"},
            stream=httpx.ByteStream(b"unpublished"),
        )

    with httpx.Client(transport=httpx.MockTransport(intercept), follow_redirects=True) as client:
        source._client = client
        # When the cohort transport reaches a non-success response.
        result = source.capture_cohort_source_prices(cohort_window())
    # Then only one attempt and a category-only receipt exist.
    assert result == PriceCaptureFailure("HTTP_ERROR", status, 0)
    assert len(calls) == 1


class AlternateWindow(CohortSourceWindow):
    """An alternate public type cannot select this closed transport."""


@pytest.mark.parametrize(
    "scope",
    [
        cohort_window().model_copy(update={"symbol": "GBP_JPY"}),
        cohort_window().model_copy(update={"scope": "legacy"}),
        cohort_window().model_copy(update={"publication_status": "published"}),
        CohortSourceWindow.model_construct(symbol="EUR_USD", day=date(2023, 1, 2)),
        AlternateWindow(
            scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=date(2023, 1, 2)
        ),
        SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3)),
    ],
)
def test_cohort_alternate_scope_refuses_before_client(
    scope: CohortSourceWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given malformed fields, a subclass or a legacy-profile object at ingress.
    source = OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")

    def unexpected_client() -> httpx.Client:
        pytest.fail("CLIENT_RESOLUTION_BEFORE_SCOPE_REFUSAL")

    monkeypatch.setattr(source, "_get_client", unexpected_client)
    # When the closed provider capability receives the unsafe object.
    result = source.capture_cohort_source_prices(scope)
    # Then the safe receipt contains no original input or body.
    assert result == PriceCaptureFailure("WINDOW_REFUSED", None, 0)


def test_cohort_caller_mutation_during_http_preserves_original_snapshot() -> None:
    # Given an original valid window and an intercepted retained client.
    source = OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")
    window = cohort_window()
    expected = window.query()
    raw = b'{"instrument":"EUR_USD","granularity":"M1","candles":[]}'

    def intercept(request: httpx.Request) -> httpx.Response:
        object.__setattr__(window, "symbol", "GBP_USD")
        object.__setattr__(window, "day", date(2025, 1, 2))
        assert request.url.path.endswith("/EUR_USD/candles")
        assert dict(request.url.params) == expected
        return httpx.Response(200, stream=httpx.ByteStream(raw))

    with httpx.Client(transport=httpx.MockTransport(intercept)) as client:
        source._client = client
        # When the caller changes its object during the wire callback.
        result = source.capture_cohort_source_prices(window)
    # Then envelope checking stays bound to the original request identity.
    assert isinstance(result, SourceBody) and result.body == raw


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"instrument":"GBP_USD","granularity":"M1","candles":[]}',
        b'{"instrument":"EUR_USD","granularity":"M1","candles":[],"extra":"dummy-token"}',
        b'{"instrument":"EUR_USD","granularity":"M1","candles":[],"extra":"dummy\\u002dtoken"}',
    ],
)
def test_cohort_secret_or_schema_failure_never_returns_body(raw: bytes) -> None:
    # Given an unsafe nonfinancial envelope on an intercepted client.
    source = OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")

    def intercept(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(raw))

    with httpx.Client(transport=httpx.MockTransport(intercept)) as client:
        source._client = client
        # When pre-retention structural and secret validation rejects the body.
        result = source.capture_cohort_source_prices(cohort_window())
    # Then only bounded safe metadata survives.
    assert isinstance(result, PriceCaptureFailure)
    assert result.http_status == 200 and result.received_bytes == len(raw)
    assert "dummy" not in repr(result) and not hasattr(result, "body")

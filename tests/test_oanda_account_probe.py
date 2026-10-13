"""Offline checks for the guarded account-candle compatibility surface."""

import json
import logging
from pathlib import Path

import httpx
import pytest
import respx

from liq.data.exceptions import UndeclaredReadError, ValidationError
from liq.data.providers.oanda_account_probe import probe_account_candle
from liq.data.service import DataService
from liq.data.settings import LiqDataSettings


@pytest.fixture
def service(tmp_path: Path) -> DataService:
    return DataService(
        settings=LiqDataSettings.model_construct(
            data_root=tmp_path,
            oanda_api_key="unused-offline-token",
            oanda_account_id="private-offline-account",
            oanda_environment="practice",
        )
    )


@respx.mock
def test_declared_probe_guards_before_one_account_get(service: DataService) -> None:
    # Given an intercepted, empty response with no invented market observations.
    def response(request: httpx.Request) -> httpx.Response:
        entry = json.loads(service.lockbox_guard.usage_log_path.read_text().splitlines()[0])
        assert entry["dataset"] == "oanda_fx"
        assert entry["purpose"] == "discovery"
        assert entry["start"] == entry["end"] == "2024-01-03"
        assert dict(request.url.params) == {
            "from": "2024-01-03T00:00:00Z",
            "to": "2024-01-03T00:00:59.999999Z",
            "price": "BA",
            "granularity": "M1",
            "units": "1",
            "smooth": "false",
            "includeFirst": "true",
        }
        assert set(request.extensions["timeout"].values()) == {30.0}
        return httpx.Response(
            200, json={"instrument": "EUR_USD", "granularity": "M1", "candles": []}
        )

    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(side_effect=response)
    # When the public service performs its declared metadata-only read.
    result = service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then absence is explicit, no prices/account identifier escape, and no pagination occurs.
    assert result.category == "EMPTY"
    assert result.raw_count == result.complete_count == 0
    assert route.call_count == 1
    assert "private-offline-account" not in repr(result)


@respx.mock
def test_probe_refuses_undeclared_read_before_transport(
    service: DataService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given no declared research purpose and no allowed HTTP route.
    def forbidden_provider(name: str) -> None:
        pytest.fail("provider resolved before declaration refusal")

    monkeypatch.setattr(service, "_get_provider", forbidden_provider)
    # When requesting the probe, the declaration guard must be the refusal.
    with pytest.raises(UndeclaredReadError):
        service.probe_oanda_account_candle(purpose=None, arm_id="offline_probe")
    # Then no request reached transport.
    assert len(respx.calls) == 0


@pytest.mark.parametrize("code", [302, 401, 429, 500])
@respx.mock
def test_probe_redacts_http_failure_and_never_redirects(service: DataService, code: int) -> None:
    # Given an error with a sensitive body and a redirect to an unapproved host.
    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(
        return_value=httpx.Response(
            code,
            text="private-offline-account unused-offline-token",
            headers={"Location": "https://unapproved.invalid/"},
        )
    )
    # When the declared probe runs.
    result = service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then it reports only status, without redirecting or exposing the body.
    assert result.category == "HTTP_ERROR"
    assert result.http_status == code
    assert route.call_count == len(respx.calls) == 1
    assert "private-offline-account" not in repr(result)
    assert "unused-offline-token" not in repr(result)


@respx.mock
def test_probe_redacts_transport_error_without_retry(service: DataService) -> None:
    # Given a wire-level failure whose text includes private request details.
    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(side_effect=httpx.ConnectError("private-offline-account unused-offline-token"))
    # When the declared probe runs.
    result = service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then the safe category is returned once, without the exception text.
    assert result.category == "TRANSPORT_ERROR"
    assert result.http_status is None
    assert route.call_count == 1
    assert "private-offline-account" not in repr(result)


@respx.mock
def test_probe_redacts_http_client_log(
    service: DataService, caplog: pytest.LogCaptureFixture
) -> None:
    # Given application logging enables HTTPX's request-level log line.
    caplog.set_level(logging.INFO, logger="httpx")
    respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(
        return_value=httpx.Response(
            200, json={"instrument": "EUR_USD", "granularity": "M1", "candles": []}
        )
    )
    # When a declared probe reaches the client.
    service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then the account identifier must not be disclosed through the client's log.
    assert "private-offline-account" not in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"instrument": "USD_JPY", "granularity": "M1", "candles": []},
        {"instrument": "EUR_USD", "granularity": "M5", "candles": []},
        {
            "instrument": "EUR_USD",
            "granularity": "M1",
            "candles": [{"time": "2024-01-03T00:01:00Z", "complete": False, "volume": 0}],
        },
    ],
)
@respx.mock
def test_probe_refuses_wrong_schema_identity_or_raw_time(service: DataService, payload) -> None:
    # Given malformed metadata without fabricated prices.
    respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(return_value=httpx.Response(200, json=payload))
    # When the response is parsed.
    result = service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then invalid source data is not labeled observed or repaired.
    assert result.category == "SCHEMA_ERROR"


@respx.mock
def test_incomplete_metadata_is_not_counted_as_accepted(service: DataService) -> None:
    # Given a metadata-only incomplete candle, with no prices invented.
    respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "instrument": "EUR_USD",
                "granularity": "M1",
                "candles": [{"time": "2024-01-03T00:00:00Z", "complete": False, "volume": 0}],
            },
        )
    )
    # When the provider marks the candle incomplete.
    result = service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then raw presence and completed acceptance remain distinct.
    assert result.category == "INCOMPLETE"
    assert result.raw_count == 1
    assert result.complete_count == 0


@respx.mock
def test_schema_exception_is_redacted(service: DataService) -> None:
    # Given malformed JSON that contains private-looking content.
    respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(return_value=httpx.Response(200, text="private-offline-account unused-offline-token"))
    # When JSON validation fails.
    result = service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then validation details never leave the provider boundary.
    assert result.category == "SCHEMA_ERROR"
    assert "private-offline-account" not in repr(result)


@respx.mock
def test_raw_duplicate_minutes_are_refused(service: DataService) -> None:
    # Given duplicate metadata only, with no invented quote prices.
    raw = {"time": "2024-01-03T00:00:00Z", "complete": False, "volume": 0}
    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/private-offline-account/instruments/EUR_USD/candles"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "instrument": "EUR_USD",
                "granularity": "M1",
                "candles": [raw, raw],
            },
        )
    )
    # When the one-minute source responds with duplicates.
    result = service.probe_oanda_account_candle(purpose="discovery", arm_id="offline_probe")
    # Then neither silent deduplication nor a second request is allowed.
    assert result.category == "SCHEMA_ERROR"
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("base_url", "account_id"),
    [
        ("https://api-fxpractice.oanda.com.unapproved.invalid", "offline-account"),
        ("https://api-fxpractice.oanda.com", "../unapproved"),
    ],
)
@respx.mock
def test_invalid_request_identity_is_refused_before_http(base_url: str, account_id: str) -> None:
    # Given an unapproved host or path-injecting account identifier.
    # When constructing the bounded request, refuse before any HTTP call.
    with httpx.Client() as client, pytest.raises(ValidationError):
        probe_account_candle(client, base_url, account_id)
    # Then no fallback or normalized alternate path is attempted.
    assert len(respx.calls) == 0

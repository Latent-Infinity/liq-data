"""Tests for the read-only Coinbase Derivatives instrument-metadata provider."""

import base64
import hashlib
import hmac
import json
import traceback
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest
import respx

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    SchemaValidationError,
)
from liq.data.providers.coinbase_derivatives import CoinbaseDerivativesProvider

SECRET = "dGVzdF9zZWNyZXRfa2V5"  # base64("test_secret_key")


@pytest.fixture
def provider() -> CoinbaseDerivativesProvider:
    return CoinbaseDerivativesProvider(
        api_key="test_key",
        api_secret=SECRET,
        passphrase="test_passphrase",
    )


def _instrument(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": "BIPZ30",
        "product_code": "BIP",
        "contract_type": "OUTRIGHT",
        "trading_state": "OPEN",
        "contract_size": 0.01,
        "activation_time": "2025-07-21T00:00:00Z",
        "expiration_time": "2030-12-20T00:00:00",
        "is_perp": True,
        "funding_interval_minutes": 60,
    }
    payload.update(overrides)
    return payload


class TestConstruction:
    @pytest.mark.parametrize("field", ["api_key", "api_secret", "passphrase"])
    def test_rejects_missing_credentials(self, field: str) -> None:
        kwargs = {"api_key": "key", "api_secret": SECRET, "passphrase": "pass"}
        kwargs[field] = ""

        with pytest.raises(ValueError, match=field):
            CoinbaseDerivativesProvider(**kwargs)

    def test_rejects_non_base64_secret(self) -> None:
        with pytest.raises(ValueError, match="base64"):
            CoinbaseDerivativesProvider("key", "not-base64!", "pass")

    def test_close_releases_an_open_client(self, provider: CoinbaseDerivativesProvider) -> None:
        client = provider._get_client()

        provider.close()

        assert client.is_closed is True
        assert provider._client is None


class TestInstrumentMetadata:
    @respx.mock
    def test_lists_and_normalizes_perpetual_instruments(
        self, provider: CoinbaseDerivativesProvider
    ) -> None:
        route = respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(200, json=[_instrument()])
        )

        instruments = provider.list_instruments(("bip", "etp"))

        assert len(instruments) == 1
        item = instruments[0]
        assert item.symbol == "BIPZ30"
        assert item.product_code == "BIP"
        assert item.contract_size == Decimal("0.01")
        assert item.activation_time == datetime(2025, 7, 21, tzinfo=UTC)
        assert item.expiration_time == datetime(2030, 12, 20, tzinfo=UTC)
        assert item.is_perp is True
        assert item.funding_interval_minutes == 60
        assert route.calls[0].request.content == b'{"product_codes":["BIP","ETP"]}'

    @respx.mock
    def test_signature_uses_exact_path_and_transmitted_body(
        self,
        provider: CoinbaseDerivativesProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("liq.data.providers.coinbase_derivatives.time.time", lambda: 1234)
        route = respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(200, json=[])
        )

        assert provider.list_instruments(("BIP",)) == ()

        request = route.calls[0].request
        body = request.content.decode()
        expected = base64.b64encode(
            hmac.new(
                base64.b64decode(SECRET),
                f"1234POST/rest/instruments{body}".encode(),
                hashlib.sha256,
            ).digest()
        ).decode()
        assert request.headers["CB-ACCESS-KEY"] == "test_key"
        assert request.headers["CB-ACCESS-PASSPHRASE"] == "test_passphrase"
        assert request.headers["CB-ACCESS-TIMESTAMP"] == "1234"
        assert request.headers["CB-ACCESS-SIGN"] == expected
        assert request.headers["Content-Type"] == "application/json"

    @respx.mock
    def test_deduplicates_product_codes_without_reordering_first_occurrence(
        self, provider: CoinbaseDerivativesProvider
    ) -> None:
        route = respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(200, json=[])
        )

        provider.list_instruments(("bip", "BIP", "etp"))

        assert json.loads(route.calls[0].request.content) == {"product_codes": ["BIP", "ETP"]}

    def test_empty_product_codes_fail_closed(self, provider: CoinbaseDerivativesProvider) -> None:
        with pytest.raises(ValueError, match="product_codes"):
            provider.list_instruments(())

    def test_string_is_not_accepted_as_a_product_code_sequence(
        self, provider: CoinbaseDerivativesProvider
    ) -> None:
        with pytest.raises(ValueError, match="product_codes"):
            provider.list_instruments("BIP")

    @respx.mock
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"symbol": ""}, "symbol"),
            ({"is_perp": None}, "is_perp"),
            ({"funding_interval_minutes": None}, "funding_interval_minutes"),
            ({"funding_interval_minutes": 0}, "funding_interval_minutes"),
            ({"contract_size": -1}, "contract_size"),
            ({"contract_size": "not-a-number"}, "contract_size"),
            ({"expiration_time": 123}, "expiration_time"),
            ({"activation_time": "not-a-time"}, "activation_time"),
        ],
    )
    def test_malformed_perpetual_metadata_fails_closed(
        self,
        provider: CoinbaseDerivativesProvider,
        overrides: dict[str, object],
        message: str,
    ) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(200, json=[_instrument(**overrides)])
        )

        with pytest.raises(SchemaValidationError, match=message):
            provider.list_instruments(("BIP",))

    @respx.mock
    def test_non_list_response_fails_closed(self, provider: CoinbaseDerivativesProvider) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(200, json={"symbol": "BIPZ30"})
        )

        with pytest.raises(SchemaValidationError, match="list"):
            provider.list_instruments(("BIP",))

    @respx.mock
    def test_non_object_instrument_fails_closed(
        self, provider: CoinbaseDerivativesProvider
    ) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(200, json=["BIPZ30"])
        )

        with pytest.raises(SchemaValidationError, match="non-object"):
            provider.list_instruments(("BIP",))

    @respx.mock
    def test_non_perpetual_instrument_may_omit_funding_interval(
        self, provider: CoinbaseDerivativesProvider
    ) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(
                200,
                json=[
                    _instrument(
                        is_perp=False,
                        funding_interval_minutes=None,
                        expiration_time=None,
                    )
                ],
            )
        )

        instrument = provider.list_instruments(("BIP",))[0]

        assert instrument.is_perp is False
        assert instrument.funding_interval_minutes is None
        assert instrument.expiration_time is None


class TestErrors:
    @respx.mock
    @pytest.mark.parametrize("status", [401, 403])
    def test_authentication_or_entitlement_failure(
        self, provider: CoinbaseDerivativesProvider, status: int
    ) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(status, json={"message": "denied"})
        )

        with pytest.raises(AuthenticationError):
            provider.list_instruments(("BIP",))

    @respx.mock
    def test_rate_limit_failure(self, provider: CoinbaseDerivativesProvider) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(429, json={"message": "slow down"})
        )

        with pytest.raises(RateLimitError):
            provider.list_instruments(("BIP",))

    @respx.mock
    def test_other_http_failure_does_not_echo_response_body(
        self, provider: CoinbaseDerivativesProvider
    ) -> None:
        secret_message = "sensitive upstream detail"
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(500, text=secret_message)
        )

        with pytest.raises(ProviderError) as raised:
            provider.list_instruments(("BIP",))

        assert secret_message not in str(raised.value)

    @respx.mock
    def test_transport_failure_is_wrapped(self, provider: CoinbaseDerivativesProvider) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            side_effect=httpx.ConnectError("offline")
        )

        with pytest.raises(ProviderError, match="request failed"):
            provider.list_instruments(("BIP",))

    @respx.mock
    def test_transport_error_does_not_expose_credential_in_traceback(
        self, provider: CoinbaseDerivativesProvider
    ) -> None:
        sentinel = "dummy-credential-sentinel"
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            side_effect=httpx.LocalProtocolError(f"Illegal header value {sentinel}")
        )

        with pytest.raises(ProviderError) as raised:
            provider.list_instruments(("BIP",))

        assert sentinel not in str(raised.value)
        assert sentinel not in "".join(traceback.format_exception(raised.value))

    @respx.mock
    def test_invalid_json_is_schema_failure(self, provider: CoinbaseDerivativesProvider) -> None:
        respx.post("https://api.exchange.fairx.net/rest/instruments").mock(
            return_value=httpx.Response(200, content=b"not-json")
        )

        with pytest.raises(SchemaValidationError, match="invalid JSON"):
            provider.list_instruments(("BIP",))


def test_provider_exposes_no_funding_or_price_read_surface() -> None:
    names = set(dir(CoinbaseDerivativesProvider))
    assert "fetch_funding_rates" not in names
    assert "fetch_bars" not in names
    assert "fetch_prices" not in names

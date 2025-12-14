"""Tests for liq.data.providers.oanda module."""

from datetime import date

import httpx
import polars as pl
import pytest
import respx

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    ValidationError,
)
from liq.data.providers.oanda import OandaProvider


@pytest.fixture
def oanda_provider() -> OandaProvider:
    """Create an OANDA provider for testing."""
    return OandaProvider(
        api_key="test_api_key",
        account_id="test_account_id",
        environment="practice",
    )


@pytest.fixture
def mock_candles_response() -> dict:
    """Sample OANDA candles response."""
    return {
        "candles": [
            {
                "time": "2024-01-15T10:00:00.000000000Z",
                "mid": {"o": "1.0850", "h": "1.0875", "l": "1.0825", "c": "1.0860"},
                "volume": 1000,
                "complete": True,
            },
            {
                "time": "2024-01-15T11:00:00.000000000Z",
                "mid": {"o": "1.0860", "h": "1.0890", "l": "1.0850", "c": "1.0885"},
                "volume": 1500,
                "complete": True,
            },
        ]
    }


@pytest.fixture
def mock_instruments_response() -> dict:
    """Sample OANDA instruments response."""
    return {
        "instruments": [
            {"name": "EUR_USD", "displayName": "Euro/US Dollar", "type": "CURRENCY"},
            {"name": "GBP_USD", "displayName": "British Pound/US Dollar", "type": "CURRENCY"},
            {"name": "USD_JPY", "displayName": "US Dollar/Japanese Yen", "type": "CURRENCY"},
        ]
    }


class TestOandaProviderCreation:
    """Tests for OandaProvider instantiation."""

    def test_create_practice_provider(self) -> None:
        provider = OandaProvider(
            api_key="key", account_id="account", environment="practice"
        )
        assert provider.name == "oanda"
        assert provider._base_url == OandaProvider.PRACTICE_URL

    def test_create_live_provider(self) -> None:
        provider = OandaProvider(
            api_key="key", account_id="account", environment="live"
        )
        assert provider._base_url == OandaProvider.LIVE_URL

    def test_supported_timeframes(self, oanda_provider: OandaProvider) -> None:
        timeframes = oanda_provider.supported_timeframes
        assert "1m" in timeframes
        assert "1h" in timeframes
        assert "1d" in timeframes


class TestOandaProviderFetchBars:
    """Tests for OandaProvider.fetch_bars method."""

    @respx.mock
    def test_fetch_bars_success(
        self,
        oanda_provider: OandaProvider,
        mock_candles_response: dict,
    ) -> None:
        # Use side_effect to return data on first call, empty on subsequent calls
        # This handles the pagination loop that would otherwise run forever with mock
        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=mock_candles_response)
            return httpx.Response(200, json={"candles": []})

        respx.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        ).mock(side_effect=mock_response)

        result = oanda_provider.fetch_bars(
            "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns

    @respx.mock
    def test_fetch_bars_empty_response(self, oanda_provider: OandaProvider) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        ).mock(return_value=httpx.Response(200, json={"candles": []}))

        result = oanda_provider.fetch_bars(
            "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @respx.mock
    def test_fetch_bars_authentication_error(self, oanda_provider: OandaProvider) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        ).mock(return_value=httpx.Response(401, json={"error": "Unauthorized"}))

        with pytest.raises(AuthenticationError, match="Invalid OANDA API credentials"):
            oanda_provider.fetch_bars(
                "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    @respx.mock
    def test_fetch_bars_rate_limit_error(self, oanda_provider: OandaProvider) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        ).mock(return_value=httpx.Response(429, json={"error": "Rate limit"}))

        with pytest.raises(RateLimitError, match="rate limit exceeded"):
            oanda_provider.fetch_bars(
                "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    def test_fetch_bars_invalid_timeframe(self, oanda_provider: OandaProvider) -> None:
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            oanda_provider.fetch_bars(
                "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="invalid"
            )

    @respx.mock
    def test_fetch_bars_skips_incomplete_candles(
        self, oanda_provider: OandaProvider
    ) -> None:
        response = {
            "candles": [
                {
                    "time": "2024-01-15T10:00:00.000000000Z",
                    "mid": {"o": "1.0850", "h": "1.0875", "l": "1.0825", "c": "1.0860"},
                    "volume": 1000,
                    "complete": True,
                },
                {
                    "time": "2024-01-15T11:00:00.000000000Z",
                    "mid": {"o": "1.0860", "h": "1.0890", "l": "1.0850", "c": "1.0885"},
                    "volume": 500,
                    "complete": False,  # Incomplete - should be skipped
                },
            ]
        }

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=response)
            return httpx.Response(200, json={"candles": []})

        respx.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        ).mock(side_effect=mock_response)

        result = oanda_provider.fetch_bars(
            "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert len(result) == 1


class TestOandaProviderListInstruments:
    """Tests for OandaProvider.list_instruments method."""

    @respx.mock
    def test_list_instruments_success(
        self,
        oanda_provider: OandaProvider,
        mock_instruments_response: dict,
    ) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/accounts/test_account_id/instruments"
        ).mock(return_value=httpx.Response(200, json=mock_instruments_response))

        result = oanda_provider.list_instruments()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "symbol" in result.columns
        assert "EUR_USD" in result["symbol"].to_list()

    @respx.mock
    def test_list_instruments_with_asset_class_filter(
        self,
        oanda_provider: OandaProvider,
        mock_instruments_response: dict,
    ) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/accounts/test_account_id/instruments"
        ).mock(return_value=httpx.Response(200, json=mock_instruments_response))

        result = oanda_provider.list_instruments(asset_class="forex")

        assert len(result) == 3

    def test_list_instruments_invalid_asset_class(
        self, oanda_provider: OandaProvider
    ) -> None:
        with pytest.raises(ProviderError, match="OANDA only supports forex"):
            oanda_provider.list_instruments(asset_class="crypto")

    @respx.mock
    def test_list_instruments_auth_error(self, oanda_provider: OandaProvider) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/accounts/test_account_id/instruments"
        ).mock(return_value=httpx.Response(401, json={"error": "Unauthorized"}))

        with pytest.raises(AuthenticationError):
            oanda_provider.list_instruments()


class TestOandaProviderValidateCredentials:
    """Tests for OandaProvider.validate_credentials method."""

    @respx.mock
    def test_validate_credentials_success(self, oanda_provider: OandaProvider) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/accounts/test_account_id"
        ).mock(return_value=httpx.Response(200, json={"account": {}}))

        assert oanda_provider.validate_credentials() is True

    @respx.mock
    def test_validate_credentials_failure(self, oanda_provider: OandaProvider) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/accounts/test_account_id"
        ).mock(return_value=httpx.Response(401, json={"error": "Unauthorized"}))

        assert oanda_provider.validate_credentials() is False

    @respx.mock
    def test_validate_credentials_network_error(
        self, oanda_provider: OandaProvider
    ) -> None:
        respx.get(
            "https://api-fxpractice.oanda.com/v3/accounts/test_account_id"
        ).mock(side_effect=httpx.ConnectError("Connection failed"))

        assert oanda_provider.validate_credentials() is False


class TestOandaProviderDataValidation:
    """Tests for OANDA data validation."""

    @respx.mock
    def test_missing_timestamp_raises_validation_error(
        self, oanda_provider: OandaProvider
    ) -> None:
        response = {
            "candles": [
                {
                    "mid": {"o": "1.0850", "h": "1.0875", "l": "1.0825", "c": "1.0860"},
                    "volume": 1000,
                    "complete": True,
                    # Missing "time" field
                }
            ]
        }

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=response)
            return httpx.Response(200, json={"candles": []})

        respx.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        ).mock(side_effect=mock_response)

        with pytest.raises(ValidationError, match="Missing timestamp"):
            oanda_provider.fetch_bars(
                "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    @respx.mock
    def test_missing_mid_prices_raises_validation_error(
        self, oanda_provider: OandaProvider
    ) -> None:
        response = {
            "candles": [
                {
                    "time": "2024-01-15T10:00:00.000000000Z",
                    "volume": 1000,
                    "complete": True,
                    # Missing "mid" field
                }
            ]
        }

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=response)
            return httpx.Response(200, json={"candles": []})

        respx.get(
            "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        ).mock(side_effect=mock_response)

        with pytest.raises(ValidationError, match="Missing mid price"):
            oanda_provider.fetch_bars(
                "EUR_USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

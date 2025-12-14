"""Tests for liq.data.providers.coinbase module."""

from datetime import date

import httpx
import polars as pl
import pytest
import respx

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from liq.data.providers.coinbase import CoinbaseProvider


@pytest.fixture
def coinbase_provider() -> CoinbaseProvider:
    """Create a Coinbase provider for testing with mock credentials.

    Note: api_secret must be a valid base64-encoded string for HMAC signing.
    """
    # Valid base64-encoded test secret (base64 of "test_secret_key_value")
    return CoinbaseProvider(
        api_key="test_api_key",
        api_secret="dGVzdF9zZWNyZXRfa2V5X3ZhbHVl",  # base64("test_secret_key_value")
        passphrase="test_passphrase",
    )


@pytest.fixture
def mock_candles_response() -> list:
    """Sample Coinbase candles response.

    Format: [timestamp, low, high, open, close, volume]
    """
    return [
        [1705312800, 41800.00, 42500.00, 42000.00, 42300.00, 100.5],
        [1705316400, 42200.00, 42800.00, 42300.00, 42700.00, 150.3],
    ]


@pytest.fixture
def mock_products_response() -> list:
    """Sample Coinbase products response."""
    return [
        {
            "id": "BTC-USD",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "display_name": "BTC/USD",
            "status": "online",
            "trading_disabled": False,
        },
        {
            "id": "ETH-USD",
            "base_currency": "ETH",
            "quote_currency": "USD",
            "display_name": "ETH/USD",
            "status": "online",
            "trading_disabled": False,
        },
        {
            "id": "SOL-USD",
            "base_currency": "SOL",
            "quote_currency": "USD",
            "display_name": "SOL/USD",
            "status": "online",
            "trading_disabled": False,
        },
    ]


class TestCoinbaseProviderCreation:
    """Tests for CoinbaseProvider instantiation."""

    def test_create_provider_with_credentials(self) -> None:
        """Test creating provider with API credentials."""
        provider = CoinbaseProvider(
            api_key="test_key",
            api_secret="dGVzdF9zZWNyZXQ=",  # base64("test_secret")
            passphrase="test_pass",
        )
        assert provider.name == "coinbase"
        assert provider._api_key == "test_key"
        assert provider._api_secret == "dGVzdF9zZWNyZXQ="
        assert provider._passphrase == "test_pass"

    def test_create_provider_without_credentials(self) -> None:
        """Test creating provider without credentials works (for public data)."""
        # Coinbase allows unauthenticated access to public endpoints
        provider = CoinbaseProvider()
        assert provider.name == "coinbase"
        assert provider._api_key is None
        assert provider._api_secret is None
        assert provider._passphrase is None
        assert provider._auth_enabled is False

    def test_create_provider_partial_credentials_disables_auth(self) -> None:
        """Test that partial credentials disable authentication."""
        # Only api_key provided - auth should be disabled
        provider = CoinbaseProvider(api_key="key")
        assert provider._auth_enabled is False

        # Only api_key and secret - auth should be disabled (missing passphrase)
        provider = CoinbaseProvider(api_key="key", api_secret="secret")
        assert provider._auth_enabled is False

    def test_create_provider_full_credentials_enables_auth(self) -> None:
        """Test that full credentials enable authentication."""
        provider = CoinbaseProvider(
            api_key="key",
            api_secret="dGVzdF9zZWNyZXQ=",
            passphrase="pass",
        )
        assert provider._auth_enabled is True

    def test_supported_timeframes(
        self, coinbase_provider: CoinbaseProvider
    ) -> None:
        """Test supported timeframes include standard intervals."""
        timeframes = coinbase_provider.supported_timeframes
        assert "1m" in timeframes
        assert "5m" in timeframes
        assert "15m" in timeframes
        assert "1h" in timeframes
        assert "1d" in timeframes


class TestCoinbaseProviderFetchBars:
    """Tests for CoinbaseProvider.fetch_bars method."""

    @respx.mock
    def test_fetch_bars_success(
        self,
        coinbase_provider: CoinbaseProvider,
        mock_candles_response: list,
    ) -> None:
        """Test fetching bar data successfully."""
        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=mock_candles_response)
            return httpx.Response(200, json=[])

        respx.get(url__regex=r".*/products/BTC-USD/candles.*").mock(
            side_effect=mock_response
        )

        result = coinbase_provider.fetch_bars(
            "BTC-USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    @respx.mock
    def test_fetch_bars_empty_response(
        self,
        coinbase_provider: CoinbaseProvider,
    ) -> None:
        """Test fetching with no data returns empty DataFrame."""
        respx.get(url__regex=r".*/products/BTC-USD/candles.*").mock(
            return_value=httpx.Response(200, json=[])
        )

        result = coinbase_provider.fetch_bars(
            "BTC-USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @respx.mock
    def test_fetch_bars_invalid_timeframe_raises(
        self,
        coinbase_provider: CoinbaseProvider,
    ) -> None:
        """Test fetching with invalid timeframe raises error."""
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            coinbase_provider.fetch_bars(
                "BTC-USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="invalid"
            )


class TestCoinbaseProviderAuthentication:
    """Tests for Coinbase API authentication."""

    @respx.mock
    def test_authentication_headers_included(
        self,
        coinbase_provider: CoinbaseProvider,
    ) -> None:
        """Test that authentication headers are included in requests."""
        route = respx.get(url__regex=r".*/products/BTC-USD/candles.*").mock(
            return_value=httpx.Response(200, json=[])
        )

        coinbase_provider.fetch_bars(
            "BTC-USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert route.called
        request = route.calls[0].request
        assert "CB-ACCESS-KEY" in request.headers
        assert "CB-ACCESS-SIGN" in request.headers
        assert "CB-ACCESS-TIMESTAMP" in request.headers
        assert "CB-ACCESS-PASSPHRASE" in request.headers

    @respx.mock
    def test_authentication_failure_raises(
        self,
        coinbase_provider: CoinbaseProvider,
    ) -> None:
        """Test authentication failure raises AuthenticationError."""
        respx.get(url__regex=r".*/products/BTC-USD/candles.*").mock(
            return_value=httpx.Response(401, json={"message": "Invalid API Key"})
        )

        with pytest.raises(AuthenticationError, match="authentication"):
            coinbase_provider.fetch_bars(
                "BTC-USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )


class TestCoinbaseProviderListInstruments:
    """Tests for CoinbaseProvider.list_instruments method."""

    @respx.mock
    def test_list_instruments_all(
        self,
        coinbase_provider: CoinbaseProvider,
        mock_products_response: list,
    ) -> None:
        """Test listing all instruments."""
        respx.get(url__regex=r".*/products$").mock(
            return_value=httpx.Response(200, json=mock_products_response)
        )

        result = coinbase_provider.list_instruments()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "symbol" in result.columns
        assert "name" in result.columns
        assert "asset_class" in result.columns

    @respx.mock
    def test_list_instruments_filters_disabled(
        self,
        coinbase_provider: CoinbaseProvider,
    ) -> None:
        """Test that disabled trading pairs are filtered out."""
        products_with_disabled = [
            {
                "id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "display_name": "BTC/USD",
                "status": "online",
                "trading_disabled": False,
            },
            {
                "id": "OLD-USD",
                "base_currency": "OLD",
                "quote_currency": "USD",
                "display_name": "OLD/USD",
                "status": "offline",
                "trading_disabled": True,
            },
        ]
        respx.get(url__regex=r".*/products$").mock(
            return_value=httpx.Response(200, json=products_with_disabled)
        )

        result = coinbase_provider.list_instruments()

        assert len(result) == 1
        assert result["symbol"][0] == "BTC-USD"


class TestCoinbaseProviderSymbolNormalization:
    """Tests for symbol normalization."""

    def test_normalize_symbol_hyphen_format(
        self, coinbase_provider: CoinbaseProvider
    ) -> None:
        """Test symbol with hyphen is unchanged."""
        assert coinbase_provider._normalize_symbol("BTC-USD") == "BTC-USD"
        assert coinbase_provider._normalize_symbol("ETH-USD") == "ETH-USD"

    def test_normalize_symbol_underscore_to_hyphen(
        self, coinbase_provider: CoinbaseProvider
    ) -> None:
        """Test symbol with underscore converts to hyphen."""
        assert coinbase_provider._normalize_symbol("BTC_USD") == "BTC-USD"
        assert coinbase_provider._normalize_symbol("ETH_USD") == "ETH-USD"

    def test_normalize_symbol_lowercase_to_uppercase(
        self, coinbase_provider: CoinbaseProvider
    ) -> None:
        """Test lowercase symbol converts to uppercase."""
        assert coinbase_provider._normalize_symbol("btc-usd") == "BTC-USD"
        assert coinbase_provider._normalize_symbol("eth_usd") == "ETH-USD"


class TestCoinbaseProviderRateLimiting:
    """Tests for rate limiting behavior."""

    @respx.mock
    def test_rate_limit_error_handling(
        self,
        coinbase_provider: CoinbaseProvider,
    ) -> None:
        """Test rate limit error is properly raised."""
        respx.get(url__regex=r".*/products/BTC-USD/candles.*").mock(
            return_value=httpx.Response(429, json={"message": "rate limit exceeded"})
        )

        with pytest.raises(RateLimitError):
            coinbase_provider.fetch_bars(
                "BTC-USD", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )


class TestCoinbaseProviderPagination:
    """Tests for pagination handling."""

    @respx.mock
    def test_pagination_multiple_requests(
        self,
        coinbase_provider: CoinbaseProvider,
        mock_candles_response: list,
    ) -> None:
        """Test pagination across multiple API requests."""
        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] <= 2:  # Return data for first 2 calls
                return httpx.Response(200, json=mock_candles_response)
            return httpx.Response(200, json=[])

        respx.get(url__regex=r".*/products/BTC-USD/candles.*").mock(
            side_effect=mock_response
        )

        result = coinbase_provider.fetch_bars(
            "BTC-USD", date(2024, 1, 1), date(2024, 1, 31), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        # Should have bars from multiple paginated requests
        assert len(result) >= 2

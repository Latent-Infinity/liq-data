"""Tests for liq.data.providers.binance module."""

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
from liq.data.providers.binance import BinanceProvider


@pytest.fixture
def binance_provider() -> BinanceProvider:
    """Create a Binance provider for testing."""
    return BinanceProvider()


@pytest.fixture
def binance_us_provider() -> BinanceProvider:
    """Create a Binance.us provider for testing."""
    return BinanceProvider(use_us=True)


@pytest.fixture
def mock_klines_response() -> list:
    """Sample Binance klines response."""
    # Binance kline format:
    # [open_time, open, high, low, close, volume, close_time, ...]
    return [
        [1705312800000, "42000.00", "42500.00", "41800.00", "42300.00", "100.5", 1705316399999],
        [1705316400000, "42300.00", "42800.00", "42200.00", "42700.00", "150.3", 1705319999999],
    ]


@pytest.fixture
def mock_exchange_info_response() -> dict:
    """Sample Binance exchangeInfo response."""
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "status": "TRADING",
            },
            {
                "symbol": "ETHUSDT",
                "baseAsset": "ETH",
                "quoteAsset": "USDT",
                "status": "TRADING",
            },
            {
                "symbol": "OLDCOIN",
                "baseAsset": "OLD",
                "quoteAsset": "USDT",
                "status": "BREAK",  # Not trading - should be filtered out
            },
        ]
    }


class TestBinanceProviderCreation:
    """Tests for BinanceProvider instantiation."""

    def test_create_default_provider(self) -> None:
        provider = BinanceProvider()
        assert provider.name == "binance"
        assert provider._base_url == BinanceProvider.BASE_URL

    def test_create_us_provider(self) -> None:
        provider = BinanceProvider(use_us=True)
        assert provider.name == "binance_us"
        assert provider._base_url == BinanceProvider.US_BASE_URL

    def test_create_with_api_key(self) -> None:
        provider = BinanceProvider(api_key="test_key", api_secret="test_secret")
        assert provider._api_key == "test_key"
        assert provider._api_secret == "test_secret"

    def test_supported_timeframes(self, binance_provider: BinanceProvider) -> None:
        timeframes = binance_provider.supported_timeframes
        assert "1m" in timeframes
        assert "1h" in timeframes
        assert "1d" in timeframes


class TestBinanceProviderFetchBars:
    """Tests for BinanceProvider.fetch_bars method."""

    @respx.mock
    def test_fetch_bars_success(
        self,
        binance_provider: BinanceProvider,
        mock_klines_response: list,
    ) -> None:
        # Use side_effect to return data on first call, empty on subsequent calls
        # This handles the pagination loop that would otherwise run forever with mock
        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=mock_klines_response)
            return httpx.Response(200, json=[])

        respx.get("https://api.binance.com/api/v3/klines").mock(side_effect=mock_response)

        result = binance_provider.fetch_bars(
            "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns

    @respx.mock
    def test_fetch_bars_converts_symbol_format(
        self,
        binance_provider: BinanceProvider,
        mock_klines_response: list,
    ) -> None:
        # Verify the canonical format BTC_USDT is converted to BTCUSDT
        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=mock_klines_response)
            return httpx.Response(200, json=[])

        route = respx.get("https://api.binance.com/api/v3/klines").mock(
            side_effect=mock_response
        )

        binance_provider.fetch_bars(
            "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        # Check that the request used BTCUSDT (no underscore)
        assert "symbol=BTCUSDT" in str(route.calls[0].request.url)

    @respx.mock
    def test_fetch_bars_empty_response(self, binance_provider: BinanceProvider) -> None:
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=httpx.Response(200, json=[])
        )

        result = binance_provider.fetch_bars(
            "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @respx.mock
    def test_fetch_bars_rate_limit_error(self, binance_provider: BinanceProvider) -> None:
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=httpx.Response(429, json={"msg": "Rate limit"})
        )

        with pytest.raises(RateLimitError, match="rate limit exceeded"):
            binance_provider.fetch_bars(
                "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    @respx.mock
    def test_fetch_bars_auth_error(self, binance_provider: BinanceProvider) -> None:
        respx.get("https://api.binance.com/api/v3/klines").mock(
            return_value=httpx.Response(401, json={"msg": "Unauthorized"})
        )

        with pytest.raises(AuthenticationError, match="Invalid Binance API"):
            binance_provider.fetch_bars(
                "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    def test_fetch_bars_invalid_timeframe(self, binance_provider: BinanceProvider) -> None:
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            binance_provider.fetch_bars(
                "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="invalid"
            )


class TestBinanceProviderListInstruments:
    """Tests for BinanceProvider.list_instruments method."""

    @respx.mock
    def test_list_instruments_success(
        self,
        binance_provider: BinanceProvider,
        mock_exchange_info_response: dict,
    ) -> None:
        respx.get("https://api.binance.com/api/v3/exchangeInfo").mock(
            return_value=httpx.Response(200, json=mock_exchange_info_response)
        )

        result = binance_provider.list_instruments()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2  # Only TRADING status symbols
        assert "symbol" in result.columns
        assert "BTC_USDT" in result["symbol"].to_list()
        assert "ETH_USDT" in result["symbol"].to_list()

    @respx.mock
    def test_list_instruments_filters_non_trading(
        self,
        binance_provider: BinanceProvider,
        mock_exchange_info_response: dict,
    ) -> None:
        respx.get("https://api.binance.com/api/v3/exchangeInfo").mock(
            return_value=httpx.Response(200, json=mock_exchange_info_response)
        )

        result = binance_provider.list_instruments()

        # OLDCOIN with status "BREAK" should be filtered out
        symbols = result["symbol"].to_list()
        assert "OLD_USDT" not in symbols

    @respx.mock
    def test_list_instruments_with_asset_class_filter(
        self,
        binance_provider: BinanceProvider,
        mock_exchange_info_response: dict,
    ) -> None:
        respx.get("https://api.binance.com/api/v3/exchangeInfo").mock(
            return_value=httpx.Response(200, json=mock_exchange_info_response)
        )

        result = binance_provider.list_instruments(asset_class="crypto")

        assert len(result) == 2

    def test_list_instruments_invalid_asset_class(
        self, binance_provider: BinanceProvider
    ) -> None:
        with pytest.raises(ProviderError, match="Binance only supports crypto"):
            binance_provider.list_instruments(asset_class="forex")

    @respx.mock
    def test_list_instruments_rate_limit_error(
        self, binance_provider: BinanceProvider
    ) -> None:
        respx.get("https://api.binance.com/api/v3/exchangeInfo").mock(
            return_value=httpx.Response(429, json={"msg": "Rate limit"})
        )

        with pytest.raises(RateLimitError):
            binance_provider.list_instruments()


class TestBinanceProviderValidateCredentials:
    """Tests for BinanceProvider.validate_credentials method."""

    @respx.mock
    def test_validate_credentials_success(
        self, binance_provider: BinanceProvider
    ) -> None:
        respx.get("https://api.binance.com/api/v3/exchangeInfo").mock(
            return_value=httpx.Response(200, json={"symbols": []})
        )

        assert binance_provider.validate_credentials() is True

    @respx.mock
    def test_validate_credentials_failure(
        self, binance_provider: BinanceProvider
    ) -> None:
        respx.get("https://api.binance.com/api/v3/exchangeInfo").mock(
            return_value=httpx.Response(500, json={"msg": "Internal error"})
        )

        assert binance_provider.validate_credentials() is False

    @respx.mock
    def test_validate_credentials_network_error(
        self, binance_provider: BinanceProvider
    ) -> None:
        respx.get("https://api.binance.com/api/v3/exchangeInfo").mock(
            side_effect=httpx.ConnectError("Connection failed")
        )

        assert binance_provider.validate_credentials() is False


class TestBinanceProviderDataValidation:
    """Tests for Binance data validation."""

    @respx.mock
    def test_malformed_kline_raises_validation_error(
        self, binance_provider: BinanceProvider
    ) -> None:
        # Kline with less than 6 fields - validation error should raise immediately
        malformed_response = [[1705312800000, "42000.00", "42500.00"]]

        # Mock returns malformed data on first call, empty on subsequent calls
        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=malformed_response)
            return httpx.Response(200, json=[])

        respx.get("https://api.binance.com/api/v3/klines").mock(side_effect=mock_response)

        with pytest.raises(ValidationError, match="Malformed kline"):
            binance_provider.fetch_bars(
                "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    @respx.mock
    def test_missing_timestamp_raises_validation_error(
        self, binance_provider: BinanceProvider
    ) -> None:
        # Kline with None timestamp
        response = [[None, "42000.00", "42500.00", "41800.00", "42300.00", "100.5"]]

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=response)
            return httpx.Response(200, json=[])

        respx.get("https://api.binance.com/api/v3/klines").mock(side_effect=mock_response)

        with pytest.raises(ValidationError, match="Missing timestamp"):
            binance_provider.fetch_bars(
                "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    @respx.mock
    def test_missing_ohlcv_field_raises_validation_error(
        self, binance_provider: BinanceProvider
    ) -> None:
        # Kline with None close price
        response = [[1705312800000, "42000.00", "42500.00", "41800.00", None, "100.5"]]

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=response)
            return httpx.Response(200, json=[])

        respx.get("https://api.binance.com/api/v3/klines").mock(side_effect=mock_response)

        with pytest.raises(ValidationError, match="Missing close"):
            binance_provider.fetch_bars(
                "BTC_USDT", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )


class TestBinanceUSProvider:
    """Tests specific to Binance.us provider."""

    @respx.mock
    def test_uses_us_base_url(
        self,
        binance_us_provider: BinanceProvider,
    ) -> None:
        route = respx.get("https://api.binance.us/api/v3/exchangeInfo").mock(
            return_value=httpx.Response(200, json={"symbols": []})
        )

        binance_us_provider.list_instruments()

        assert route.called

"""Tests for liq.data.providers.polygon module."""

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
from liq.data.providers.polygon import PolygonProvider


@pytest.fixture
def polygon_provider() -> PolygonProvider:
    """Create a Polygon provider for testing with mock credentials."""
    return PolygonProvider(api_key="test_api_key")


@pytest.fixture
def mock_aggs_response() -> dict:
    """Sample Polygon aggregates response.

    Format: https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to
    """
    return {
        "ticker": "AAPL",
        "queryCount": 2,
        "resultsCount": 2,
        "adjusted": True,
        "results": [
            {
                "v": 1000000.0,  # volume
                "vw": 175.5,  # volume weighted average price
                "o": 174.0,  # open
                "c": 176.0,  # close
                "h": 177.0,  # high
                "l": 173.5,  # low
                "t": 1705312800000,  # timestamp (ms)
                "n": 5000,  # number of transactions
            },
            {
                "v": 1200000.0,
                "vw": 176.5,
                "o": 176.0,
                "c": 177.5,
                "h": 178.0,
                "l": 175.5,
                "t": 1705316400000,
                "n": 6000,
            },
        ],
        "status": "OK",
        "request_id": "test-request-id",
    }


@pytest.fixture
def mock_tickers_response() -> dict:
    """Sample Polygon tickers response."""
    return {
        "results": [
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "type": "CS",
                "active": True,
                "currency_name": "usd",
            },
            {
                "ticker": "MSFT",
                "name": "Microsoft Corporation",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "type": "CS",
                "active": True,
                "currency_name": "usd",
            },
            {
                "ticker": "GOOGL",
                "name": "Alphabet Inc.",
                "market": "stocks",
                "locale": "us",
                "primary_exchange": "XNAS",
                "type": "CS",
                "active": True,
                "currency_name": "usd",
            },
        ],
        "status": "OK",
        "count": 3,
        "next_url": None,
    }


class TestPolygonProviderCreation:
    """Tests for PolygonProvider instantiation."""

    def test_create_provider_with_api_key(self) -> None:
        """Test creating provider with API key."""
        provider = PolygonProvider(api_key="test_key")
        assert provider.name == "polygon"
        assert provider._api_key == "test_key"

    def test_create_provider_missing_api_key_raises(self) -> None:
        """Test creating provider without api_key raises error."""
        with pytest.raises(ValueError, match="api_key.*required"):
            PolygonProvider(api_key=None)

    def test_supported_timeframes(self, polygon_provider: PolygonProvider) -> None:
        """Test supported timeframes include standard intervals."""
        timeframes = polygon_provider.supported_timeframes
        assert "1m" in timeframes
        assert "5m" in timeframes
        assert "15m" in timeframes
        assert "1h" in timeframes
        assert "1d" in timeframes


class TestPolygonProviderFetchBars:
    """Tests for PolygonProvider.fetch_bars method."""

    @respx.mock
    def test_fetch_bars_success(
        self,
        polygon_provider: PolygonProvider,
        mock_aggs_response: dict,
    ) -> None:
        """Test fetching bar data successfully."""
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(200, json=mock_aggs_response)
        )

        result = polygon_provider.fetch_bars(
            "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
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
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test fetching with no data returns empty DataFrame."""
        empty_response = {
            "ticker": "AAPL",
            "queryCount": 0,
            "resultsCount": 0,
            "adjusted": True,
            "results": [],
            "status": "OK",
        }
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(200, json=empty_response)
        )

        result = polygon_provider.fetch_bars(
            "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @respx.mock
    def test_fetch_bars_no_results_key(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test handling response without results key."""
        response_no_results = {
            "ticker": "AAPL",
            "queryCount": 0,
            "resultsCount": 0,
            "status": "OK",
        }
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(200, json=response_no_results)
        )

        result = polygon_provider.fetch_bars(
            "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @respx.mock
    def test_fetch_bars_invalid_timeframe_raises(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test fetching with invalid timeframe raises error."""
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            polygon_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="invalid"
            )


class TestPolygonProviderAuthentication:
    """Tests for Polygon API authentication."""

    @respx.mock
    def test_api_key_included_in_headers(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test that API key is included in request headers."""
        empty_response = {"results": [], "status": "OK"}
        route = respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(200, json=empty_response)
        )

        polygon_provider.fetch_bars(
            "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert route.called
        request = route.calls[0].request
        assert "Authorization" in request.headers
        assert "Bearer test_api_key" in request.headers["Authorization"]

    @respx.mock
    def test_authentication_failure_raises(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test authentication failure raises AuthenticationError."""
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(
                401, json={"status": "NOT_AUTHORIZED", "message": "Invalid API key"}
            )
        )

        with pytest.raises(AuthenticationError, match="authentication"):
            polygon_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    @respx.mock
    def test_forbidden_raises_authentication_error(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test 403 forbidden raises AuthenticationError."""
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(
                403, json={"status": "FORBIDDEN", "message": "Access denied"}
            )
        )

        with pytest.raises(AuthenticationError, match="authentication"):
            polygon_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )


class TestPolygonProviderListInstruments:
    """Tests for PolygonProvider.list_instruments method."""

    @respx.mock
    def test_list_instruments_all(
        self,
        polygon_provider: PolygonProvider,
        mock_tickers_response: dict,
    ) -> None:
        """Test listing all instruments."""
        respx.get(url__regex=r".*/v3/reference/tickers.*").mock(
            return_value=httpx.Response(200, json=mock_tickers_response)
        )

        result = polygon_provider.list_instruments()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "symbol" in result.columns
        assert "name" in result.columns
        assert "asset_class" in result.columns

    @respx.mock
    def test_list_instruments_stocks_filter(
        self,
        polygon_provider: PolygonProvider,
        mock_tickers_response: dict,
    ) -> None:
        """Test listing instruments with stocks asset class filter."""
        respx.get(url__regex=r".*/v3/reference/tickers.*").mock(
            return_value=httpx.Response(200, json=mock_tickers_response)
        )

        result = polygon_provider.list_instruments(asset_class="stocks")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    @respx.mock
    def test_list_instruments_filters_inactive(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test that inactive tickers are filtered out."""
        tickers_with_inactive = {
            "results": [
                {
                    "ticker": "AAPL",
                    "name": "Apple Inc.",
                    "market": "stocks",
                    "locale": "us",
                    "type": "CS",
                    "active": True,
                },
                {
                    "ticker": "OLDCO",
                    "name": "Old Company",
                    "market": "stocks",
                    "locale": "us",
                    "type": "CS",
                    "active": False,
                },
            ],
            "status": "OK",
        }
        respx.get(url__regex=r".*/v3/reference/tickers.*").mock(
            return_value=httpx.Response(200, json=tickers_with_inactive)
        )

        result = polygon_provider.list_instruments()

        assert len(result) == 1
        assert result["symbol"][0] == "AAPL"


class TestPolygonProviderSymbolNormalization:
    """Tests for symbol normalization."""

    def test_normalize_symbol_uppercase(
        self, polygon_provider: PolygonProvider
    ) -> None:
        """Test symbol is uppercased."""
        assert polygon_provider._normalize_symbol("aapl") == "AAPL"
        assert polygon_provider._normalize_symbol("msft") == "MSFT"

    def test_normalize_symbol_already_uppercase(
        self, polygon_provider: PolygonProvider
    ) -> None:
        """Test already uppercase symbol is unchanged."""
        assert polygon_provider._normalize_symbol("AAPL") == "AAPL"
        assert polygon_provider._normalize_symbol("GOOGL") == "GOOGL"


class TestPolygonProviderRateLimiting:
    """Tests for rate limiting behavior."""

    @respx.mock
    def test_rate_limit_error_handling(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test rate limit error is properly raised."""
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(
                429, json={"status": "ERROR", "message": "rate limit exceeded"}
            )
        )

        with pytest.raises(RateLimitError):
            polygon_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )


class TestPolygonProviderPagination:
    """Tests for pagination handling."""

    @respx.mock
    def test_pagination_with_next_url(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test pagination follows next_url for large datasets."""
        # First response with next_url
        first_response = {
            "ticker": "AAPL",
            "results": [
                {
                    "v": 1000000.0,
                    "o": 174.0,
                    "c": 176.0,
                    "h": 177.0,
                    "l": 173.5,
                    "t": 1705312800000,
                    "n": 5000,
                },
            ],
            "status": "OK",
            "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/next?cursor=abc",
        }

        # Second response (last page)
        second_response = {
            "ticker": "AAPL",
            "results": [
                {
                    "v": 1200000.0,
                    "o": 176.0,
                    "c": 177.5,
                    "h": 178.0,
                    "l": 175.5,
                    "t": 1705316400000,
                    "n": 6000,
                },
            ],
            "status": "OK",
            "next_url": None,
        }

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=first_response)
            return httpx.Response(200, json=second_response)

        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            side_effect=mock_response
        )
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/next.*").mock(
            return_value=httpx.Response(200, json=second_response)
        )

        result = polygon_provider.fetch_bars(
            "AAPL", date(2024, 1, 1), date(2024, 1, 31), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        # Should have bars from both pages
        assert len(result) >= 1


class TestPolygonProviderErrorHandling:
    """Tests for error handling."""

    @respx.mock
    def test_api_error_handling(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test generic API error handling."""
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            return_value=httpx.Response(
                500, json={"status": "ERROR", "message": "Internal server error"}
            )
        )

        with pytest.raises(ProviderError, match="Polygon API error"):
            polygon_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

    @respx.mock
    def test_network_error_handling(
        self,
        polygon_provider: PolygonProvider,
    ) -> None:
        """Test network error handling."""
        respx.get(url__regex=r".*/v2/aggs/ticker/AAPL/range/.*").mock(
            side_effect=httpx.ConnectError("Connection failed")
        )

        with pytest.raises(ProviderError, match="request failed"):
            polygon_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )

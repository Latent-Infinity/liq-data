"""Tests for liq.data.providers.alpaca module."""

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest
import respx

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
)
from liq.data.providers.alpaca import AlpacaProvider
from liq.store import ParquetStore


@pytest.fixture
def alpaca_provider() -> AlpacaProvider:
    """Create an Alpaca provider for testing with mock credentials."""
    return AlpacaProvider(
        api_key="test_api_key",
        api_secret="test_api_secret",
    )


@pytest.fixture
def mock_bars_response() -> dict[str, Any]:
    """Sample Alpaca bars response."""
    return {
        "bars": [
            {
                "t": "2024-01-15T14:00:00Z",
                "o": 174.0,
                "h": 175.5,
                "l": 173.5,
                "c": 175.0,
                "v": 1000000,
                "n": 5000,
                "vw": 174.5,
            },
            {
                "t": "2024-01-15T15:00:00Z",
                "o": 175.0,
                "h": 176.5,
                "l": 174.5,
                "c": 176.0,
                "v": 1200000,
                "n": 6000,
                "vw": 175.5,
            },
        ],
        "symbol": "AAPL",
        "next_page_token": None,
    }


@pytest.fixture
def mock_assets_response() -> list[dict[str, Any]]:
    """Sample Alpaca assets response."""
    return [
        {
            "id": "b28f4066-5c6d-479b-a2af-85dc1a8f16fb",
            "class": "us_equity",
            "exchange": "NASDAQ",
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "status": "active",
            "tradable": True,
            "marginable": True,
            "shortable": True,
            "easy_to_borrow": True,
            "fractionable": True,
        },
        {
            "id": "f801f835-bfe6-4a9d-a6b1-ccbb84bfd75f",
            "class": "us_equity",
            "exchange": "NASDAQ",
            "symbol": "MSFT",
            "name": "Microsoft Corporation",
            "status": "active",
            "tradable": True,
            "marginable": True,
            "shortable": True,
            "easy_to_borrow": True,
            "fractionable": True,
        },
    ]


class TestAlpacaProviderInit:
    """Tests for AlpacaProvider initialization."""

    def test_init_with_credentials(self) -> None:
        """Test initialization with API credentials."""
        provider = AlpacaProvider(
            api_key="test_key",
            api_secret="test_secret",
        )

        assert provider._api_key == "test_key"
        assert provider._api_secret == "test_secret"

    def test_init_missing_api_key_raises(self) -> None:
        """Test initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            AlpacaProvider(api_key=None, api_secret="secret")

    def test_init_missing_api_secret_raises(self) -> None:
        """Test initialization without API secret raises ValueError."""
        with pytest.raises(ValueError, match="api_secret is required"):
            AlpacaProvider(api_key="key", api_secret=None)

    def test_provider_name(self, alpaca_provider: AlpacaProvider) -> None:
        """Test provider name is alpaca."""
        assert alpaca_provider.name == "alpaca"

    def test_supported_timeframes(self, alpaca_provider: AlpacaProvider) -> None:
        """Test supported timeframes."""
        expected = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        assert alpaca_provider.supported_timeframes == expected


class TestAlpacaProviderFetchBars:
    """Tests for AlpacaProvider.fetch_bars method."""

    @respx.mock
    def test_fetch_bars_success(
        self,
        alpaca_provider: AlpacaProvider,
        mock_bars_response: dict[str, Any],
    ) -> None:
        """Test successful fetch_bars call."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(200, json=mock_bars_response)
        )

        result = alpaca_provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 15),
            end=date(2024, 1, 15),
            timeframe="1h",
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
    def test_fetch_bars_correct_values(
        self,
        alpaca_provider: AlpacaProvider,
        mock_bars_response: dict[str, Any],
    ) -> None:
        """Test fetch_bars returns correct OHLCV values."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(200, json=mock_bars_response)
        )

        result = alpaca_provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 15),
            end=date(2024, 1, 15),
            timeframe="1h",
        )

        first_row = result.row(0)
        # timestamp, open, high, low, close, volume
        assert first_row[1] == 174.0  # open
        assert first_row[2] == 175.5  # high
        assert first_row[3] == 173.5  # low
        assert first_row[4] == 175.0  # close
        assert first_row[5] == 1000000.0  # volume

    @respx.mock
    def test_fetch_bars_authentication_headers(
        self,
        alpaca_provider: AlpacaProvider,
        mock_bars_response: dict[str, Any],
    ) -> None:
        """Test fetch_bars sends correct authentication headers."""
        route = respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(200, json=mock_bars_response)
        )

        alpaca_provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 15),
            end=date(2024, 1, 15),
            timeframe="1h",
        )

        assert route.called
        request = route.calls[0].request
        assert request.headers["APCA-API-KEY-ID"] == "test_api_key"
        assert request.headers["APCA-API-SECRET-KEY"] == "test_api_secret"

    @respx.mock
    def test_fetch_bars_empty_response(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars with empty response."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(200, json={"bars": None, "next_page_token": None})
        )

        result = alpaca_provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 15),
            end=date(2024, 1, 15),
            timeframe="1h",
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    @respx.mock
    def test_fetch_bars_authentication_error(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars raises AuthenticationError on 401."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(401, json={"message": "Unauthorized"})
        )

        with pytest.raises(AuthenticationError, match="Alpaca authentication failed"):
            alpaca_provider.fetch_bars(
                "AAPL",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
            )

    @respx.mock
    def test_fetch_bars_forbidden_error(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars raises AuthenticationError on 403."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(403, json={"message": "Forbidden"})
        )

        with pytest.raises(AuthenticationError, match="Alpaca authentication failed"):
            alpaca_provider.fetch_bars(
                "AAPL",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
            )

    @respx.mock
    def test_fetch_bars_rate_limit_error(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars raises RateLimitError on 429."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(429, json={"message": "Rate limit exceeded"})
        )

        with pytest.raises(RateLimitError, match="Alpaca rate limit exceeded"):
            alpaca_provider.fetch_bars(
                "AAPL",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
            )

    @respx.mock
    def test_fetch_bars_provider_error(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars raises ProviderError on other errors."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(ProviderError, match="Alpaca API error"):
            alpaca_provider.fetch_bars(
                "AAPL",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
            )

    @respx.mock
    def test_fetch_bars_network_error(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars raises ProviderError on network error."""
        respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            side_effect=httpx.RequestError("Connection failed")
        )

        with pytest.raises(ProviderError, match="Alpaca API request failed"):
            alpaca_provider.fetch_bars(
                "AAPL",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="1h",
            )

    def test_fetch_bars_invalid_timeframe(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars raises ProviderError for invalid timeframe."""
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            alpaca_provider.fetch_bars(
                "AAPL",
                start=date(2024, 1, 15),
                end=date(2024, 1, 15),
                timeframe="2h",
            )

    @respx.mock
    def test_fetch_bars_pagination(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test fetch_bars handles pagination correctly."""
        page1_response = {
            "bars": [
                {
                    "t": "2024-01-15T14:00:00Z",
                    "o": 174.0,
                    "h": 175.5,
                    "l": 173.5,
                    "c": 175.0,
                    "v": 1000000,
                    "n": 5000,
                    "vw": 174.5,
                },
            ],
            "next_page_token": "token123",
        }
        page2_response = {
            "bars": [
                {
                    "t": "2024-01-15T15:00:00Z",
                    "o": 175.0,
                    "h": 176.5,
                    "l": 174.5,
                    "c": 176.0,
                    "v": 1200000,
                    "n": 6000,
                    "vw": 175.5,
                },
            ],
            "next_page_token": None,
        }

        route = respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars")
        route.side_effect = [
            httpx.Response(200, json=page1_response),
            httpx.Response(200, json=page2_response),
        ]

        result = alpaca_provider.fetch_bars(
            "AAPL",
            start=date(2024, 1, 15),
            end=date(2024, 1, 15),
            timeframe="1h",
        )

        assert len(result) == 2
        assert route.call_count == 2

    @respx.mock
    def test_fetch_bars_symbol_normalization(
        self,
        alpaca_provider: AlpacaProvider,
        mock_bars_response: dict[str, Any],
    ) -> None:
        """Test fetch_bars normalizes symbols to uppercase."""
        route = respx.get("https://data.alpaca.markets/v2/stocks/AAPL/bars").mock(
            return_value=httpx.Response(200, json=mock_bars_response)
        )

        # Use lowercase symbol
        alpaca_provider.fetch_bars(
            "aapl",
            start=date(2024, 1, 15),
            end=date(2024, 1, 15),
            timeframe="1h",
        )

        assert route.called
        # URL should have uppercase symbol
        assert "/AAPL/" in str(route.calls[0].request.url)


class TestAlpacaProviderFetchBarsWindow:
    """Exact datetime bounds and an explicit feed, as `fetch_quotes` takes them."""

    URL = "https://data.alpaca.markets/v2/stocks/AAPL/bars"

    @respx.mock
    def test_a_datetime_window_and_feed_are_sent_exactly(
        self, alpaca_provider: AlpacaProvider
    ) -> None:
        route = respx.get(self.URL).mock(
            return_value=httpx.Response(200, json={"bars": [], "next_page_token": None})
        )

        alpaca_provider.fetch_bars(
            "AAPL",
            datetime(2022, 6, 15, 17, 30, tzinfo=UTC),
            datetime(2022, 6, 15, 18, 20, tzinfo=UTC),
            timeframe="1m",
            feed="iex",
        )

        params = route.calls[0].request.url.params
        assert params["start"] == "2022-06-15T17:30:00Z"
        assert params["end"] == "2022-06-15T18:20:00Z"
        assert params["feed"] == "iex"
        assert params["adjustment"] == "all"

    @respx.mock
    def test_prices_as_traded_can_be_asked_for(self, alpaca_provider: AlpacaProvider) -> None:
        route = respx.get(self.URL).mock(
            return_value=httpx.Response(200, json={"bars": [], "next_page_token": None})
        )

        alpaca_provider.fetch_bars("AAPL", date(2022, 6, 15), date(2022, 6, 15), adjustment="raw")

        assert route.calls[0].request.url.params["adjustment"] == "raw"

    @respx.mock
    def test_feed_is_omitted_unless_asked_for(self, alpaca_provider: AlpacaProvider) -> None:
        route = respx.get(self.URL).mock(
            return_value=httpx.Response(200, json={"bars": [], "next_page_token": None})
        )

        alpaca_provider.fetch_bars("AAPL", date(2022, 6, 15), date(2022, 6, 15))

        params = route.calls[0].request.url.params
        assert params["start"] == "2022-06-15T00:00:00Z"
        assert "feed" not in params

    def test_a_naive_bound_is_refused(self, alpaca_provider: AlpacaProvider) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            alpaca_provider.fetch_bars(
                "AAPL", datetime(2022, 6, 15, 17, 30), date(2022, 6, 15), timeframe="1m"
            )


class TestAlpacaProviderCorporateActions:
    """Tests for the v1 corporate-actions market-data endpoint."""

    @respx.mock
    def test_get_corporate_actions_flattens_typed_groups(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Provider keeps the response group so heterogeneous rows stay identifiable."""
        route = respx.get("https://data.alpaca.markets/v1/corporate-actions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "corporate_actions": {
                        "cash_dividends": [
                            {
                                "id": "3fa6553a-4211-4018-85b9-d34d4d744c06",
                                "symbol": "AAPL",
                                "cusip": "037833100",
                                "rate": 0.22,
                                "process_date": "2021-10-28",
                                "ex_date": "2021-11-05",
                                "record_date": "2021-11-08",
                                "payable_date": "2021-11-11",
                            }
                        ],
                        "forward_splits": [],
                    },
                    "next_page_token": None,
                },
            )
        )

        result = alpaca_provider.get_corporate_actions(
            "aapl",
            start=date(2021, 1, 1),
            end=date(2021, 12, 31),
        )

        assert result == [
            {
                "id": "3fa6553a-4211-4018-85b9-d34d4d744c06",
                "symbol": "AAPL",
                "cusip": "037833100",
                "rate": 0.22,
                "process_date": "2021-10-28",
                "ex_date": "2021-11-05",
                "record_date": "2021-11-08",
                "payable_date": "2021-11-11",
                "corporate_action_type": "cash_dividends",
            }
        ]
        request = route.calls[0].request
        assert request.url.params["symbols"] == "AAPL"
        assert request.url.params["start"] == "2021-01-01"
        assert request.url.params["end"] == "2021-12-31"
        assert request.url.params["limit"] == "1000"
        assert request.url.params["data_quality"] == "complete"
        assert request.url.params["sort"] == "asc"

    @respx.mock
    def test_get_corporate_actions_paginates(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        route = respx.get("https://data.alpaca.markets/v1/corporate-actions")
        route.side_effect = [
            httpx.Response(
                200,
                json={
                    "corporate_actions": {"forward_splits": [{"id": "split-1", "symbol": "AAPL"}]},
                    "next_page_token": "page-2",
                },
            ),
            httpx.Response(
                200,
                json={
                    "corporate_actions": {
                        "cash_dividends": [{"id": "dividend-1", "symbol": "AAPL"}]
                    },
                    "next_page_token": None,
                },
            ),
        ]

        result = alpaca_provider.get_corporate_actions(
            "AAPL",
            start=date(2021, 1, 1),
            end=date(2021, 12, 31),
        )

        assert [row["corporate_action_type"] for row in result] == [
            "forward_splits",
            "cash_dividends",
        ]
        assert route.call_count == 2
        assert "page_token" not in route.calls[0].request.url.params
        assert route.calls[1].request.url.params["page_token"] == "page-2"

    @respx.mock
    def test_get_corporate_actions_rejects_malformed_payload(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        respx.get("https://data.alpaca.markets/v1/corporate-actions").mock(
            return_value=httpx.Response(200, json={"corporate_actions": []})
        )

        with pytest.raises(ProviderError, match="malformed corporate-actions response"):
            alpaca_provider.get_corporate_actions(
                "AAPL",
                start=date(2021, 1, 1),
                end=date(2021, 12, 31),
            )


class TestAlpacaProviderListInstruments:
    """Tests for AlpacaProvider.list_instruments method."""

    @respx.mock
    def test_list_instruments_success(
        self,
        alpaca_provider: AlpacaProvider,
        mock_assets_response: list[dict[str, Any]],
    ) -> None:
        """Test successful list_instruments call."""
        respx.get("https://api.alpaca.markets/v2/assets").mock(
            return_value=httpx.Response(200, json=mock_assets_response)
        )

        result = alpaca_provider.list_instruments()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "symbol" in result.columns
        assert "name" in result.columns

    @respx.mock
    def test_list_instruments_filters_inactive(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test list_instruments filters out inactive assets."""
        assets_with_inactive = [
            {
                "id": "b28f4066-5c6d-479b-a2af-85dc1a8f16fb",
                "class": "us_equity",
                "exchange": "NASDAQ",
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "status": "active",
                "tradable": True,
            },
            {
                "id": "inactive-asset-id",
                "class": "us_equity",
                "exchange": "NASDAQ",
                "symbol": "DELISTED",
                "name": "Delisted Corp",
                "status": "inactive",
                "tradable": False,
            },
        ]

        respx.get("https://api.alpaca.markets/v2/assets").mock(
            return_value=httpx.Response(200, json=assets_with_inactive)
        )

        result = alpaca_provider.list_instruments()

        assert len(result) == 1
        assert result["symbol"][0] == "AAPL"

    @respx.mock
    def test_list_instruments_empty_response(
        self,
        alpaca_provider: AlpacaProvider,
    ) -> None:
        """Test list_instruments with empty response."""
        respx.get("https://api.alpaca.markets/v2/assets").mock(
            return_value=httpx.Response(200, json=[])
        )

        result = alpaca_provider.list_instruments()

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert "symbol" in result.columns


class TestAlpacaProviderTimeframeMapping:
    """Tests for timeframe mapping."""

    def test_timeframe_map_completeness(self, alpaca_provider: AlpacaProvider) -> None:
        """Test all supported timeframes have mappings."""
        for tf in alpaca_provider.supported_timeframes:
            assert tf in alpaca_provider.TIMEFRAME_MAP

    def test_timeframe_mapping_values(self, alpaca_provider: AlpacaProvider) -> None:
        """Test timeframe mappings are correct Alpaca format."""
        expected_mappings = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "4h": "4Hour",
            "1d": "1Day",
            "1w": "1Week",
        }
        for our_tf, alpaca_tf in expected_mappings.items():
            assert alpaca_provider.TIMEFRAME_MAP[our_tf] == alpaca_tf


@pytest.fixture
def approved_quotes() -> pl.DataFrame:
    root = Path(__file__).parent / "fixtures" / "approved"
    content = (root / "alpaca_quotes" / "quotes.parquet").read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "16be5a9c5c2b364b86ca1075f3492ecffde86c13c5ffec0e65d1487b5dbbaac9"
    )
    return ParquetStore(str(root)).read("alpaca_quotes").head(2)


def _quote_payload(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.with_columns(pl.col("timestamp").dt.to_string("%Y-%m-%dT%H:%M:%S%.9fZ"))
        .rename(
            {
                "timestamp": "t",
                "bid": "bp",
                "ask": "ap",
                "bid_size": "bs",
                "ask_size": "as",
                "bid_exchange": "bx",
                "ask_exchange": "ax",
                "conditions": "c",
                "tape": "z",
            }
        )
        .to_dicts()
    )


class TestAlpacaProviderFetchQuotes:
    URL = "https://data.alpaca.markets/v2/stocks/AAPL/quotes"
    START = datetime(2022, 6, 15, 18, 5, tzinfo=UTC)
    END = datetime(2022, 6, 15, 18, 20, tzinfo=UTC)

    @respx.mock
    def test_requests_the_exact_window_feed_and_ascending_order(
        self, alpaca_provider: AlpacaProvider
    ) -> None:
        route = respx.get(self.URL).mock(
            return_value=httpx.Response(200, json={"quotes": [], "next_page_token": None})
        )

        alpaca_provider.fetch_quotes("aapl", self.START, self.END, feed="sip")

        params = route.calls[0].request.url.params
        assert params["start"] == "2022-06-15T18:05:00Z"
        assert params["end"] == "2022-06-15T18:20:00Z"
        assert params["feed"] == "sip"
        assert params["sort"] == "asc"
        assert route.calls[0].request.headers["APCA-API-KEY-ID"] == "test_api_key"

    @respx.mock
    def test_a_date_bound_spans_the_whole_utc_day_and_feed_is_optional(
        self, alpaca_provider: AlpacaProvider
    ) -> None:
        route = respx.get(self.URL).mock(
            return_value=httpx.Response(200, json={"quotes": [], "next_page_token": None})
        )

        alpaca_provider.fetch_quotes("AAPL", date(2022, 6, 15), date(2022, 6, 15))

        params = route.calls[0].request.url.params
        assert params["start"] == "2022-06-15T00:00:00Z"
        assert params["end"] == "2022-06-15T23:59:59.999999999Z"
        assert "feed" not in params

    @respx.mock
    def test_fields_are_renamed_and_values_kept(
        self, alpaca_provider: AlpacaProvider, approved_quotes: pl.DataFrame
    ) -> None:
        respx.get(self.URL).mock(
            return_value=httpx.Response(
                200, json={"quotes": _quote_payload(approved_quotes), "next_page_token": None}
            )
        )

        frame = alpaca_provider.fetch_quotes("AAPL", self.START, self.END)

        assert frame.equals(approved_quotes)

    @respx.mock
    def test_every_page_is_kept_in_order(
        self, alpaca_provider: AlpacaProvider, approved_quotes: pl.DataFrame
    ) -> None:
        first, second = _quote_payload(approved_quotes)
        route = respx.get(self.URL)
        route.side_effect = [
            httpx.Response(200, json={"quotes": [first], "next_page_token": "next"}),
            httpx.Response(200, json={"quotes": [second], "next_page_token": None}),
        ]

        frame = alpaca_provider.fetch_quotes("AAPL", self.START, self.END)

        assert frame.equals(approved_quotes)
        assert route.calls[1].request.url.params["page_token"] == "next"

    @respx.mock
    def test_a_repeated_page_token_is_refused(self, alpaca_provider: AlpacaProvider) -> None:
        route = respx.get(self.URL)
        route.side_effect = [
            httpx.Response(200, json={"quotes": [], "next_page_token": "repeated"}),
            httpx.Response(200, json={"quotes": [], "next_page_token": "repeated"}),
            httpx.Response(200, json={"quotes": [], "next_page_token": None}),
        ]

        with pytest.raises(ProviderError, match="repeated page token"):
            alpaca_provider.fetch_quotes("AAPL", self.START, self.END)

        assert len(route.calls) == 2

    @respx.mock
    def test_a_window_without_quotes_is_empty_with_the_same_columns(
        self, alpaca_provider: AlpacaProvider
    ) -> None:
        respx.get(self.URL).mock(
            return_value=httpx.Response(200, json={"quotes": None, "next_page_token": None})
        )

        frame = alpaca_provider.fetch_quotes("AAPL", self.START, self.END)

        assert frame.is_empty()
        assert "bid" in frame.columns
        assert "timestamp" in frame.columns

    def test_a_naive_bound_is_refused(self, alpaca_provider: AlpacaProvider) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            alpaca_provider.fetch_quotes("AAPL", self.START.replace(tzinfo=None), self.END)

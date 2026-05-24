"""Tests for liq.data.providers.tradestation module."""

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
from liq.data.providers.tradestation import TradeStationProvider


@pytest.fixture
def tradestation_provider() -> TradeStationProvider:
    """Create a TradeStation provider for testing with mock credentials.

    Sets ``retry_base_delay_s=0.0`` so backoff-retry tests don't actually wait.
    """
    return TradeStationProvider(
        client_id="test_client_id",
        client_secret="test_client_secret",
        refresh_token="test_refresh_token",
        retry_base_delay_s=0.0,
    )


@pytest.fixture
def mock_bars_response() -> dict:
    """Sample TradeStation bars response."""
    # TradeStation GetBars response format
    return {
        "Bars": [
            {
                "High": "150.50",
                "Low": "149.00",
                "Open": "149.50",
                "Close": "150.25",
                "TimeStamp": "2024-01-15T09:30:00Z",
                "TotalVolume": "1000000",
            },
            {
                "High": "151.00",
                "Low": "150.00",
                "Open": "150.25",
                "Close": "150.75",
                "TimeStamp": "2024-01-15T10:30:00Z",
                "TotalVolume": "1200000",
            },
        ]
    }


@pytest.fixture
def mock_token_response() -> dict:
    """Sample OAuth2 token response."""
    return {
        "access_token": "new_access_token",
        "refresh_token": "new_refresh_token",
        "expires_in": 1200,
        "token_type": "Bearer",
    }


@pytest.fixture
def mock_symbols_response() -> dict:
    """Sample TradeStation symbols response."""
    return {
        "Symbols": [
            {
                "Symbol": "AAPL",
                "Description": "Apple Inc.",
                "Exchange": "NASDAQ",
                "AssetType": "STOCK",
            },
            {
                "Symbol": "MSFT",
                "Description": "Microsoft Corporation",
                "Exchange": "NASDAQ",
                "AssetType": "STOCK",
            },
            {
                "Symbol": "ESZ24",
                "Description": "E-mini S&P 500 Dec 2024",
                "Exchange": "CME",
                "AssetType": "FUTURE",
            },
        ]
    }


class TestTradeStationProviderCreation:
    """Tests for TradeStationProvider instantiation."""

    def test_create_provider_with_credentials(self) -> None:
        """Test creating provider with OAuth2 credentials."""
        provider = TradeStationProvider(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token",
        )
        assert provider.name == "tradestation"
        assert provider._client_id == "test_client_id"
        assert provider._client_secret == "test_client_secret"
        assert provider._refresh_token == "test_refresh_token"

    def test_create_provider_missing_credentials_raises(self) -> None:
        """Test creating provider without credentials raises error."""
        with pytest.raises(ValueError, match="client_id.*required"):
            TradeStationProvider(
                client_id=None,
                client_secret="secret",
                refresh_token="token",
            )

    def test_supported_timeframes(self, tradestation_provider: TradeStationProvider) -> None:
        """Test supported timeframes include standard intervals."""
        timeframes = tradestation_provider.supported_timeframes
        assert "1m" in timeframes
        assert "5m" in timeframes
        assert "1h" in timeframes
        assert "1d" in timeframes


class TestTradeStationProviderFetchBars:
    """Tests for TradeStationProvider.fetch_bars method."""

    @respx.mock
    def test_fetch_bars_stocks_success(
        self,
        tradestation_provider: TradeStationProvider,
        mock_bars_response: dict,
        mock_token_response: dict,
    ) -> None:
        """Test fetching stock bar data."""
        # Mock OAuth2 token endpoint
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )

        # Mock bars endpoint - use side_effect to return data then empty
        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=mock_bars_response)
            return httpx.Response(200, json={"Bars": []})

        respx.get(url__regex=r".*/marketdata/barcharts/AAPL.*").mock(side_effect=mock_response)

        result = tradestation_provider.fetch_bars(
            "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    @respx.mock
    def test_fetch_bars_futures_success(
        self,
        tradestation_provider: TradeStationProvider,
        mock_bars_response: dict,
        mock_token_response: dict,
    ) -> None:
        """Test fetching futures bar data."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(200, json=mock_bars_response)
            return httpx.Response(200, json={"Bars": []})

        respx.get(url__regex=r".*/marketdata/barcharts/ESZ24.*").mock(side_effect=mock_response)

        result = tradestation_provider.fetch_bars(
            "ESZ24", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    @respx.mock
    def test_fetch_bars_invalid_timeframe_raises(
        self,
        tradestation_provider: TradeStationProvider,
    ) -> None:
        """Test fetching with invalid timeframe raises error."""
        with pytest.raises(ProviderError, match="Unsupported timeframe"):
            tradestation_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="invalid"
            )


class TestTradeStationProviderOAuth2:
    """Tests for TradeStation OAuth2 authentication."""

    @respx.mock
    def test_oauth2_token_refresh(
        self,
        tradestation_provider: TradeStationProvider,
        mock_token_response: dict,
    ) -> None:
        """Test OAuth2 token refresh flow."""
        route = respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )

        # Trigger token refresh
        tradestation_provider._ensure_authenticated()

        assert route.called
        assert tradestation_provider._access_token == "new_access_token"

    @respx.mock
    def test_oauth2_token_refresh_failure_raises(
        self,
        tradestation_provider: TradeStationProvider,
    ) -> None:
        """Test OAuth2 token refresh failure raises AuthenticationError."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(401, json={"error": "invalid_grant"})
        )

        with pytest.raises(AuthenticationError, match="OAuth2"):
            tradestation_provider._ensure_authenticated()


class TestTradeStationProviderListInstruments:
    """Tests for TradeStationProvider.list_instruments method."""

    @respx.mock
    def test_list_instruments_all(
        self,
        tradestation_provider: TradeStationProvider,
        mock_symbols_response: dict,
        mock_token_response: dict,
    ) -> None:
        """Test listing all instruments."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        respx.get(url__regex=r".*/marketdata/symbols.*").mock(
            return_value=httpx.Response(200, json=mock_symbols_response)
        )

        result = tradestation_provider.list_instruments()

        assert isinstance(result, pl.DataFrame)
        # Both STOCK and FUTURE searches return data, so we get symbols from both
        assert len(result) >= 3
        assert "symbol" in result.columns
        assert "name" in result.columns
        assert "asset_class" in result.columns

    @respx.mock
    def test_list_instruments_stocks_only(
        self,
        tradestation_provider: TradeStationProvider,
        mock_symbols_response: dict,
        mock_token_response: dict,
    ) -> None:
        """Test listing only stock instruments."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        respx.get(url__regex=r".*/marketdata/symbols.*").mock(
            return_value=httpx.Response(200, json=mock_symbols_response)
        )

        result = tradestation_provider.list_instruments(asset_class="stocks")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2  # AAPL and MSFT
        assert all(result["asset_class"] == "STOCK")

    @respx.mock
    def test_list_instruments_futures_only(
        self,
        tradestation_provider: TradeStationProvider,
        mock_symbols_response: dict,
        mock_token_response: dict,
    ) -> None:
        """Test listing only futures instruments."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        respx.get(url__regex=r".*/marketdata/symbols.*").mock(
            return_value=httpx.Response(200, json=mock_symbols_response)
        )

        result = tradestation_provider.list_instruments(asset_class="futures")

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1  # ESZ24
        assert all(result["asset_class"] == "FUTURE")


class TestTradeStationProviderSymbolNormalization:
    """Tests for symbol normalization."""

    def test_normalize_stock_symbol(self, tradestation_provider: TradeStationProvider) -> None:
        """Test stock symbol normalization (should be unchanged)."""
        assert tradestation_provider._normalize_symbol("AAPL") == "AAPL"
        assert tradestation_provider._normalize_symbol("MSFT") == "MSFT"

    def test_normalize_futures_symbol(self, tradestation_provider: TradeStationProvider) -> None:
        """Test futures symbol normalization."""
        # Continuous futures symbol
        assert tradestation_provider._normalize_symbol("ES") == "@ES"
        # Specific contract
        assert tradestation_provider._normalize_symbol("ESZ24") == "ESZ24"


class TestTradeStationProviderRateLimiting:
    """Tests for rate limiting behavior."""

    @respx.mock
    def test_rate_limit_error_handling(
        self,
        tradestation_provider: TradeStationProvider,
        mock_token_response: dict,
    ) -> None:
        """Test rate limit error is properly raised."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        respx.get(url__regex=r".*/marketdata/barcharts/.*").mock(
            return_value=httpx.Response(429, json={"error": "rate_limit_exceeded"})
        )

        with pytest.raises(RateLimitError):
            tradestation_provider.fetch_bars(
                "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
            )


class TestTradeStationProviderPagination:
    """Tests for pagination handling."""

    @respx.mock
    def test_pagination_multiple_requests(
        self,
        tradestation_provider: TradeStationProvider,
        mock_bars_response: dict,
        mock_token_response: dict,
    ) -> None:
        """Test pagination across multiple API requests."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )

        call_count = {"n": 0}

        def mock_response(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] <= 2:  # Return data for first 2 calls
                return httpx.Response(200, json=mock_bars_response)
            return httpx.Response(200, json={"Bars": []})

        respx.get(url__regex=r".*/marketdata/barcharts/.*").mock(side_effect=mock_response)

        result = tradestation_provider.fetch_bars(
            "AAPL", date(2024, 1, 1), date(2024, 1, 31), timeframe="1h"
        )

        assert isinstance(result, pl.DataFrame)
        # Should have bars from multiple paginated requests
        assert len(result) >= 2


class TestTradeStationProviderIntradayLimits:
    """Tests for the 500k-minute-per-request intraday limit handling."""

    def test_safe_bars_caps_intraday_at_minute_equivalents(
        self,
        tradestation_provider: TradeStationProvider,
    ) -> None:
        """15m bars: 500_000/15 = 33_333 < MAX_BARS_PER_REQUEST (57_600)."""
        assert tradestation_provider._safe_bars_per_request("15m") == 500_000 // 15
        assert tradestation_provider._safe_bars_per_request("30m") == 500_000 // 30
        assert tradestation_provider._safe_bars_per_request("1h") == 500_000 // 60
        assert tradestation_provider._safe_bars_per_request("4h") == 500_000 // 240

    def test_safe_bars_keeps_max_for_fine_intraday_under_limit(
        self,
        tradestation_provider: TradeStationProvider,
    ) -> None:
        """1m bars: 500_000/1 = 500_000 > MAX_BARS so MAX_BARS wins; 5m similar."""
        assert tradestation_provider._safe_bars_per_request("1m") == 57_600
        assert tradestation_provider._safe_bars_per_request("5m") == 57_600

    def test_safe_bars_uses_max_for_daily_and_weekly(
        self,
        tradestation_provider: TradeStationProvider,
    ) -> None:
        """Daily/Weekly bars are not subject to the minute-equivalent cap."""
        assert tradestation_provider._safe_bars_per_request("1d") == 57_600
        assert tradestation_provider._safe_bars_per_request("1w") == 57_600

    @respx.mock
    def test_fetch_bars_15m_uses_chunked_barsback(
        self,
        tradestation_provider: TradeStationProvider,
        mock_bars_response: dict,
        mock_token_response: dict,
    ) -> None:
        """At 15m the request's ``barsback`` param must be 33333, not 57600."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        captured_params: list[dict[str, str]] = []

        def _capture(request: httpx.Request) -> httpx.Response:
            captured_params.append(dict(request.url.params))
            # Return empty after first call to terminate pagination.
            if len(captured_params) == 1:
                return httpx.Response(200, json=mock_bars_response)
            return httpx.Response(200, json={"Bars": []})

        respx.get(url__regex=r".*/marketdata/barcharts/SPY.*").mock(side_effect=_capture)
        tradestation_provider.fetch_bars(
            "SPY", date(2024, 1, 1), date(2024, 1, 31), timeframe="15m"
        )
        assert captured_params
        assert captured_params[0]["barsback"] == str(500_000 // 15)


class TestTokenPersistGuard:
    """Regression: ensure the conftest autouse fixture blocks .env writes.

    Without the guard, a mocked OAuth refresh response containing
    ``refresh_token=<mock value>`` would trigger
    ``persist_env_value("TRADESTATION_REFRESH_TOKEN", <mock value>)``,
    overwriting the developer's real refresh token. This actually happened
    in dev — the guard exists so it cannot happen again.
    """

    def test_persist_env_value_is_a_no_op_during_tests(
        self, tmp_path: "object"
    ) -> None:  # type: ignore[name-defined]
        # The autouse fixture replaces the attribute on the module. Importing
        # and calling it must do nothing — specifically, it must NOT write to
        # any path we pass it.
        from pathlib import Path

        from liq.data.settings import persist_env_value

        target = Path(str(tmp_path)) / ".env"
        persist_env_value("ANY_KEY", "any_value", env_path=target)
        assert not target.exists(), (
            "persist_env_value must be a no-op during tests "
            "(conftest._block_real_env_writes is required)"
        )

    @respx.mock
    def test_token_rotation_during_fetch_does_not_persist(
        self,
        tradestation_provider: TradeStationProvider,
        mock_bars_response: dict,
        mock_token_response: dict,
        tmp_path: "object",
    ) -> None:  # type: ignore[name-defined]
        """Even a full fetch with mock-rotated token leaves no .env on disk."""
        from pathlib import Path

        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        respx.get(url__regex=r".*/marketdata/barcharts/.*").mock(
            return_value=httpx.Response(200, json={"Bars": []})
        )
        canary = Path(str(tmp_path)) / "would_have_been_env"
        # The provider does not know about `canary`; the test asserts that
        # neither it nor the cwd .env gets written. (Cwd check kept light to
        # avoid scribbling in repo-root .env even if the guard ever regresses.)
        before_cwd_env = Path(".env").read_bytes() if Path(".env").exists() else None
        tradestation_provider.fetch_bars(
            "AAPL", date(2024, 1, 15), date(2024, 1, 15), timeframe="1h"
        )
        after_cwd_env = Path(".env").read_bytes() if Path(".env").exists() else None
        assert not canary.exists()
        assert before_cwd_env == after_cwd_env, (
            "Test must not modify cwd .env via the token-rotation persist path."
        )


class TestTradeStationProviderRateLimitBackoff:
    """Tests for the retry-with-backoff behavior on RateLimitError."""

    @respx.mock
    def test_rate_limit_retries_then_succeeds(
        self,
        tradestation_provider: TradeStationProvider,
        mock_bars_response: dict,
        mock_token_response: dict,
    ) -> None:
        """429 on first call, 200 on retry → fetch_bars succeeds."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        call_count = {"n": 0}

        def _flaky(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return httpx.Response(429, json={"error": "rate_limit_exceeded"})
            if call_count["n"] == 2:
                return httpx.Response(200, json=mock_bars_response)
            return httpx.Response(200, json={"Bars": []})

        respx.get(url__regex=r".*/marketdata/barcharts/.*").mock(side_effect=_flaky)
        result = tradestation_provider.fetch_bars(
            "AAPL", date(2024, 1, 1), date(2024, 1, 31), timeframe="1h"
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) >= 1
        # First attempt 429, second attempt 200, third attempt empty-terminate.
        assert call_count["n"] >= 2

    @respx.mock
    def test_rate_limit_exhausts_retries_then_raises(
        self,
        tradestation_provider: TradeStationProvider,
        mock_token_response: dict,
    ) -> None:
        """Persistent 429 → after retry_max_attempts+1 calls, RateLimitError raises."""
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=mock_token_response)
        )
        call_count = {"n": 0}

        def _always_429(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(429, json={"error": "rate_limit_exceeded"})

        respx.get(url__regex=r".*/marketdata/barcharts/.*").mock(side_effect=_always_429)
        with pytest.raises(RateLimitError):
            tradestation_provider.fetch_bars(
                "AAPL", date(2024, 1, 1), date(2024, 1, 31), timeframe="1h"
            )
        # Default retry_max_attempts=3 → 4 total attempts.
        assert call_count["n"] == tradestation_provider._retry_max_attempts + 1

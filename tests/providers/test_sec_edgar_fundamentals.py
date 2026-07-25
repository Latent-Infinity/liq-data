"""SEC EDGAR fundamentals provider tests (transport mocked with respx).

The companyfacts payload is a real committed fixture; only the HTTP transport
is mocked, following the existing ``test_sec_edgar`` pattern.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.providers.sec_edgar_fundamentals import (
    COMPANYFACTS_URL,
    SECEdgarFundamentalsProvider,
)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "edgar"

TICKERS_JSON = {
    "0": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA Corp"},
}


def _companyfacts(symbol: str) -> dict:
    return json.loads((FIXTURES / f"{symbol}.companyfacts.json").read_text())


def _provider() -> SECEdgarFundamentalsProvider:
    return SECEdgarFundamentalsProvider("Test Contact test@example.com", min_interval_seconds=0.0)


class TestConstruction:
    def test_requires_user_agent(self) -> None:
        with pytest.raises(ConfigurationError, match="user_agent"):
            SECEdgarFundamentalsProvider("")


class TestFetchFundamentals:
    @respx.mock
    def test_fetch_msft_snapshot(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(COMPANYFACTS_URL.format(cik10="0000789019")).mock(
            return_value=httpx.Response(200, json=_companyfacts("MSFT"))
        )
        snap = _provider().fetch_fundamentals("MSFT", as_of=date(2026, 3, 31))
        assert snap is not None
        assert snap.symbol == "MSFT"
        assert snap.cik == "0000789019"
        assert snap.latest is not None
        assert snap.latest.fiscal_year == 2025
        assert snap.latest.fcf is not None and snap.latest.fcf > 0

    @respx.mock
    def test_unknown_ticker_returns_none(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        snap = _provider().fetch_fundamentals("NOPE", as_of=date(2026, 3, 31))
        assert snap is None

    @respx.mock
    def test_resolve_cik_and_companyfacts_url(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        facts_route = respx.get(COMPANYFACTS_URL.format(cik10="0001045810")).mock(
            return_value=httpx.Response(200, json=_companyfacts("NVDA"))
        )
        provider = _provider()
        assert provider.resolve_cik("NVDA") == "0001045810"
        snap = provider.fetch_fundamentals("NVDA", as_of=date(2026, 3, 31))
        assert facts_route.called
        assert snap is not None and snap.latest is not None
        assert snap.latest.net_debt is not None and snap.latest.net_debt < 0


class TestTransportErrors:
    @respx.mock
    def test_http_error_raises_provider_error(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(COMPANYFACTS_URL.format(cik10="0000789019")).mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(ProviderError):
            _provider().fetch_fundamentals("MSFT", as_of=date(2026, 3, 31))

    @respx.mock
    def test_rate_limit_raises(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(COMPANYFACTS_URL.format(cik10="0000789019")).mock(
            return_value=httpx.Response(429)
        )
        with pytest.raises(RateLimitError):
            _provider().fetch_fundamentals("MSFT", as_of=date(2026, 3, 31))

    @respx.mock
    def test_client_is_reused_across_calls(self) -> None:
        respx.get(COMPANYFACTS_URL.format(cik10="0000789019")).mock(
            return_value=httpx.Response(200, json=_companyfacts("MSFT"))
        )
        provider = _provider()
        provider.fetch_companyfacts("0000789019")
        client = provider._get_client()
        provider.fetch_companyfacts("0000789019")
        assert provider._get_client() is client

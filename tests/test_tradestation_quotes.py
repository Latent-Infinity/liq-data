"""TradeStationProvider.fetch_quote_snapshots: batched snapshot quotes."""

from __future__ import annotations

import httpx
import polars as pl
import pytest
import respx

from liq.data.providers.tradestation import TradeStationProvider


@pytest.fixture
def provider() -> TradeStationProvider:
    return TradeStationProvider(
        client_id="test_client_id",
        client_secret="test_client_secret",
        refresh_token="test_refresh_token",
    )


@pytest.fixture
def token_response() -> dict:
    return {"access_token": "test_access_token", "expires_in": 1200}


def _quote(symbol: str, last: str, high: str, prev: str) -> dict:
    return {
        "Symbol": symbol,
        "Last": last,
        "High": high,
        "Low": "1.00",
        "PreviousClose": prev,
    }


class TestFetchQuotes:
    @respx.mock
    def test_returns_frame_with_float_columns(
        self, provider: TradeStationProvider, token_response: dict
    ) -> None:
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=token_response)
        )
        respx.get(url__regex=r".*/marketdata/quotes/AAPL,MSFT.*").mock(
            return_value=httpx.Response(
                200,
                json={
                    "Quotes": [
                        _quote("AAPL", "150.25", "152.00", "149.00"),
                        _quote("MSFT", "410.10", "415.00", "408.00"),
                    ]
                },
            )
        )

        result = provider.fetch_quote_snapshots(["AAPL", "MSFT"])

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2
        assert set(result.columns) >= {"symbol", "last", "high", "previous_close"}
        row = result.filter(pl.col("symbol") == "AAPL").to_dicts()[0]
        assert row["last"] == 150.25
        assert row["high"] == 152.00
        assert row["previous_close"] == 149.00

    @respx.mock
    def test_batches_requests_of_at_most_100_symbols(
        self, provider: TradeStationProvider, token_response: dict
    ) -> None:
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=token_response)
        )
        seen_urls: list[str] = []

        def responder(request: httpx.Request) -> httpx.Response:
            seen_urls.append(str(request.url))
            batch = str(request.url).split("/quotes/")[1].split("?")[0]
            symbols = batch.split(",")
            return httpx.Response(
                200,
                json={"Quotes": [_quote(s, "10.0", "11.0", "10.5") for s in symbols]},
            )

        respx.get(url__regex=r".*/marketdata/quotes/.*").mock(side_effect=responder)

        symbols = [f"SYM{i:03d}" for i in range(250)]
        result = provider.fetch_quote_snapshots(symbols)

        assert result.height == 250
        assert len(seen_urls) == 3  # 100 + 100 + 50

    @respx.mock
    def test_skips_unpriced_quotes_never_fabricates(
        self, provider: TradeStationProvider, token_response: dict
    ) -> None:
        respx.post("https://signin.tradestation.com/oauth/token").mock(
            return_value=httpx.Response(200, json=token_response)
        )
        respx.get(url__regex=r".*/marketdata/quotes/.*").mock(
            return_value=httpx.Response(
                200,
                json={
                    "Quotes": [
                        _quote("AAPL", "150.25", "152.00", "149.00"),
                        {"Symbol": "HALTED", "Last": None, "High": None},
                    ]
                },
            )
        )

        result = provider.fetch_quote_snapshots(["AAPL", "HALTED"])

        assert result.height == 1
        assert result["symbol"].to_list() == ["AAPL"]

    def test_empty_symbol_list_rejected(self, provider: TradeStationProvider) -> None:
        with pytest.raises(ValueError):
            provider.fetch_quote_snapshots([])

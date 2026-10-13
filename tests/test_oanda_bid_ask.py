"""Offline contract tests for OANDA historical bid/ask candles."""

from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest
import respx

from liq.data.exceptions import ProviderError, ValidationError
from liq.data.providers.oanda import OandaProvider


@pytest.fixture
def provider() -> OandaProvider:
    return OandaProvider(api_key="test-token", account_id="test-account")


@respx.mock
def test_bid_ask_request_is_explicit_and_empty_response_is_typed(
    provider: OandaProvider,
) -> None:
    # Given an empty provider response, which contains no invented prices.
    route = respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        return_value=httpx.Response(200, json={"candles": []})
    )

    # When historical bid/ask candles are requested.
    bars = provider.fetch_bid_ask_bars("EUR_USD", date(2024, 1, 3), date(2024, 1, 3), "1m")

    # Then both price components are requested and the empty schema is stable.
    assert route.called
    assert route.calls[0].request.url.params["price"] == "BA"
    assert bars.columns == [
        "timestamp",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "volume",
    ]
    assert bars.height == 0


@respx.mock
def test_bid_ask_response_rejects_missing_price_side(provider: OandaProvider) -> None:
    # Given a structurally incomplete response, with no fabricated quote values.
    respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        return_value=httpx.Response(
            200,
            json={
                "candles": [{key: value for key, value in PUBLISHED_CANDLE.items() if key != "bid"}]
            },
        )
    )

    # When the bid/ask path parses it.
    with pytest.raises(ValidationError, match="Missing bid price"):
        provider.fetch_bid_ask_bars("EUR_USD", date(2016, 1, 3), date(2016, 1, 3), "1m")


# Third-party published example claiming an OANDA EUR_USD M5 response, 2016-01-03
# 22:00 UTC. Author: hootnot, posted 2017-06-25:
# https://github.com/hootnot/oanda-api-v20/issues/75#issuecomment-310893761
# One excerpted candle only; not an authenticated capture or admitted research fixture.
PUBLISHED_CANDLE = {
    "complete": True,
    "bid": {"h": "1.08699", "c": "1.08676", "l": "1.08674", "o": "1.08699"},
    "mid": {"h": "1.08746", "c": "1.08726", "l": "1.08724", "o": "1.08743"},
    "volume": 6,
    "time": "2016-01-03T22:00:00.000000000Z",
    "ask": {"h": "1.08796", "c": "1.08776", "l": "1.08774", "o": "1.08787"},
}


@respx.mock
def test_bid_ask_parses_published_candle_without_replacing_midpoint(
    provider: OandaProvider,
) -> None:
    # Given one published candle and an empty remainder of the requested day.
    route = respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        side_effect=[
            httpx.Response(200, json={"candles": [PUBLISHED_CANDLE]}),
            httpx.Response(200, json={"candles": []}),
        ]
    )

    # When bid/ask candles are loaded through the provider.
    bars = provider.fetch_bid_ask_bars("EUR_USD", date(2016, 1, 3), date(2016, 1, 3), "5m")

    # Then exact published sides survive parsing; the midpoint is not substituted.
    assert route.calls[0].request.url.params["price"] == "BA"
    assert bars.height == 1
    row = bars.row(0, named=True)
    assert row["timestamp"] == datetime(2016, 1, 3, 22, tzinfo=UTC)
    assert row["bid_close"] == Decimal("1.08676")
    assert row["ask_close"] == Decimal("1.08776")


@respx.mock
def test_midpoint_callers_keep_their_original_price_basis(provider: OandaProvider) -> None:
    # Given the same third-party published candle, used only as an API-shape fixture.
    route = respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        side_effect=[
            httpx.Response(200, json={"candles": [PUBLISHED_CANDLE]}),
            httpx.Response(200, json={"candles": []}),
        ]
    )

    # When the existing OHLCV API is used.
    bars = provider.fetch_bars("EUR_USD", date(2016, 1, 3), date(2016, 1, 3), "5m")

    # Then it still requests midpoints and returns the original schema.
    assert route.calls[0].request.url.params["price"] == "M"
    assert bars.columns == ["timestamp", "open", "high", "low", "close", "volume"]
    assert bars["close"][0] == Decimal("1.08726")


@respx.mock
def test_error_response_body_is_not_exposed(provider: OandaProvider) -> None:
    respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        return_value=httpx.Response(500, text="private-provider-response-body")
    )
    with pytest.raises(ProviderError) as caught:
        provider.fetch_bid_ask_bars("EUR_USD", date(2024, 1, 3), date(2024, 1, 3))
    assert "500" in str(caught.value)
    assert "private-provider-response-body" not in str(caught.value)


def test_account_projection_exposes_metadata_not_published_price_values() -> None:
    from dataclasses import asdict

    from liq.data.providers.oanda_account_probe import summarize_account_candle

    # Given the already-approved public candle unchanged. Its source is M5;
    # this test envelope exercises API shape only and is NOT an M1 capture.
    response = httpx.Response(
        200,
        json={
            "instrument": "EUR_USD",
            "granularity": "M1",
            "candles": [PUBLISHED_CANDLE],
        },
    )
    start = datetime(2016, 1, 3, 22, tzinfo=UTC)
    # When the pure parser validates the source values and projects metadata.
    result = summarize_account_candle(response, start)
    # Then only counts, timestamp, and response category escape the parser.
    assert asdict(result) == {
        "category": "OBSERVED",
        "http_status": 200,
        "raw_count": 1,
        "complete_count": 1,
        "timestamps": (start,),
    }


@pytest.mark.parametrize("field", ["price_type", "volume_missing", "volume_boolean", "envelope"])
@respx.mock
def test_bid_ask_rejects_malformed_payload(provider: OandaProvider, field: str) -> None:
    # Given published prices with a malformed type or missing metadata, never new prices.
    candle = deepcopy(PUBLISHED_CANDLE)
    if field == "price_type":
        candle["bid"]["o"] = True
    elif field == "volume_missing":
        del candle["volume"]
    elif field == "volume_boolean":
        candle["volume"] = True
    payload = {} if field == "envelope" else {"candles": [candle]}
    route = respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        side_effect=[httpx.Response(200, json=payload), httpx.Response(200, json={"candles": []})]
    )
    # When parsing invalid bid/ask data, refuse it before further pagination.
    with pytest.raises(ValidationError):
        provider.fetch_bid_ask_bars("EUR_USD", date(2016, 1, 3), date(2016, 1, 3))
    assert route.call_count == 1


@respx.mock
def test_non_boolean_completion_flag_is_refused(provider: OandaProvider) -> None:
    # Change only metadata structure; preserve the published price values.
    malformed = {**PUBLISHED_CANDLE, "complete": "false"}
    respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        side_effect=[
            httpx.Response(200, json={"candles": [malformed]}),
            httpx.Response(200, json={"candles": []}),
        ]
    )
    with pytest.raises(ValidationError):
        provider.fetch_bid_ask_bars("EUR_USD", date(2016, 1, 3), date(2016, 1, 3), "5m")


@pytest.mark.parametrize("duplicate", [False, True])
@respx.mock
def test_quote_page_refuses_out_of_scope_or_duplicate_timestamps(
    provider: OandaProvider,
    duplicate: bool,
) -> None:
    candles = [PUBLISHED_CANDLE, PUBLISHED_CANDLE] if duplicate else [PUBLISHED_CANDLE]
    day = date(2016, 1, 3) if duplicate else date(2016, 1, 4)
    route = respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        side_effect=[
            httpx.Response(200, json={"candles": candles}),
            httpx.Response(200, json={"candles": []}),
        ],
    )
    with pytest.raises(ValidationError, match="timestamp"):
        provider.fetch_bid_ask_bars("EUR_USD", day, day, "5m")
    assert route.call_count == 1

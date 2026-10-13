"""Decode OANDA historical candles without conflating mid and bid/ask prices."""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr
from pydantic import ValidationError as PayloadValidationError

from liq.data.exceptions import AuthenticationError, ProviderError, RateLimitError, ValidationError

PriceComponents = Literal["M", "BA"]


class _Prices(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    open: Annotated[StrictStr, Field(pattern=r"^\d+(?:\.\d+)?$")] = Field(alias="o")
    high: Annotated[StrictStr, Field(pattern=r"^\d+(?:\.\d+)?$")] = Field(alias="h")
    low: Annotated[StrictStr, Field(pattern=r"^\d+(?:\.\d+)?$")] = Field(alias="l")
    close: Annotated[StrictStr, Field(pattern=r"^\d+(?:\.\d+)?$")] = Field(alias="c")


class _BidAskCandle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    time: StrictStr
    complete: StrictBool
    volume: Annotated[StrictInt, Field(ge=0)]
    bid: _Prices | None = None
    ask: _Prices | None = None


class _BidAskEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    candles: list[_BidAskCandle]


def fetch_candle_chunk(
    client: httpx.Client,
    base_url: str,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    granularity: str,
    price_components: PriceComponents,
) -> list[dict[str, Any]]:
    """Fetch and decode one OANDA candle page for an explicit price basis."""
    from_time = start_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    to_time = end_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    url = (
        f"{base_url}/v3/instruments/{symbol}/candles"
        f"?from={from_time}&to={to_time}&granularity={granularity}"
        f"&price={price_components}"
    )
    response = client.get(url)
    if response.status_code == 401:
        raise AuthenticationError("Invalid OANDA API credentials")
    if response.status_code == 429:
        raise RateLimitError("OANDA API rate limit exceeded")
    if response.status_code != 200:
        raise ProviderError(f"OANDA API error: {response.status_code}")

    if price_components == "BA":
        return _decode_bid_ask(response, start_dt, end_dt)

    bars: list[dict[str, Any]] = []
    for candle in response.json().get("candles", []):
        if not candle.get("complete", False):
            continue
        if "time" not in candle:
            raise ValidationError("Missing timestamp in candle data")
        timestamp_str = candle["time"]
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        mid = _price_side(candle, "mid", timestamp_str)
        bars.append(
            {
                "timestamp": timestamp,
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": float(candle.get("volume", 0)),
            }
        )
    return bars


_OHLC_KEYS = {"open": "o", "high": "h", "low": "l", "close": "c"}


def _decode_bid_ask(
    response: httpx.Response,
    start: datetime,
    end: datetime,
) -> list[dict[str, datetime | Decimal]]:
    """Parse strict bid/ask payloads, rejecting malformed data without echoing prices."""
    try:
        payload = _BidAskEnvelope.model_validate_json(response.content)
    except PayloadValidationError:
        raise ValidationError("Invalid OANDA bid/ask candle payload") from None
    return _bid_ask_rows(payload, start, end)


def _bid_ask_rows(
    payload: _BidAskEnvelope,
    start: datetime,
    end: datetime,
) -> list[dict[str, datetime | Decimal]]:
    """Decode already parsed candles without repeating boundary validation."""
    rows: list[dict[str, datetime | Decimal]] = []
    previous: datetime | None = None
    for candle in payload.candles:
        if not candle.complete:
            continue
        try:
            timestamp = datetime.fromisoformat(candle.time.replace("Z", "+00:00"))
        except ValueError:
            raise ValidationError("Invalid OANDA bid/ask candle timestamp") from None
        if (
            timestamp.tzinfo is None
            or not start <= timestamp <= end
            or (previous is not None and timestamp <= previous)
        ):
            raise ValidationError("OANDA bid/ask candle timestamp order or bounds invalid")
        previous = timestamp
        row: dict[str, datetime | Decimal] = {
            "timestamp": timestamp,
            "volume": Decimal(candle.volume),
        }
        for side, prices in (("bid", candle.bid), ("ask", candle.ask)):
            if prices is None:
                raise ValidationError(f"Missing {side} price data")
            for name in _OHLC_KEYS:
                value = Decimal(getattr(prices, name))
                if not value.is_finite() or value <= 0:
                    raise ValidationError("Invalid OANDA bid/ask price domain")
                row[f"{side}_{name}"] = value
        rows.append(row)
    return rows


def _price_side(candle: dict[str, Any], side: str, timestamp: str) -> dict[str, str]:
    if side not in candle:
        raise ValidationError(f"Missing {side} price data at {timestamp}")
    prices: dict[str, str] = candle[side]
    for key in _OHLC_KEYS.values():
        if key not in prices:
            raise ValidationError(f"Missing OHLC field '{key}' at {timestamp}")
    return prices

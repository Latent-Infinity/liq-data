"""FRED (Federal Reserve Economic Data) provider.

Wraps the ``fredapi`` package to fetch macroeconomic time series and
expose them in the standard OHLCV bar schema. FRED series carry a single
value per observation date (no OHLC microstructure), so each bar has
``open == high == low == close == value`` and ``volume == 0``. This keeps
downstream pipelines (storage, alignment, resampling) uniform across
providers without inventing macro-specific schemas.

Examples of useful FRED series for trading-strategy regime features:
  - ``T10Y2Y``     — 10y minus 2y Treasury constant-maturity spread.
  - ``BAMLH0A0HYM2`` — ICE BofA US High Yield index option-adjusted spread.
  - ``DTWEXBGS``   — Trade-weighted US dollar index (broad).
  - ``ICSA``       — Initial unemployment insurance claims (weekly).
  - ``INDPRO``     — Industrial Production Index (monthly).
  - ``NAPM``       — ISM Manufacturing PMI (monthly).
  - ``UNRATE``     — Civilian unemployment rate (monthly).

Auth: ``FRED_API_KEY`` environment variable (free key from
https://fred.stlouisfed.org/docs/api/api_key.html).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, Protocol

import polars as pl

from liq.data.providers.base import PRICE_DTYPE, VOLUME_DTYPE, BaseProvider


class _FREDClientLike(Protocol):
    """Minimal interface FREDProvider needs from a FRED client."""

    def get_series(
        self,
        series_id: str,
        observation_start: date | str | None = ...,
        observation_end: date | str | None = ...,
    ) -> Any: ...  # pragma: no cover - protocol


def _default_client_factory(api_key: str) -> _FREDClientLike:
    """Build a real ``fredapi.Fred`` client. Imported lazily so the rest
    of the package is usable without the optional ``fredapi`` install."""
    from fredapi import Fred

    return Fred(api_key=api_key)


_EMPTY_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "open": PRICE_DTYPE,
    "high": PRICE_DTYPE,
    "low": PRICE_DTYPE,
    "close": PRICE_DTYPE,
    "volume": VOLUME_DTYPE,
}


class FREDProvider(BaseProvider):
    """Federal Reserve Economic Data provider.

    Symbols are FRED series IDs (e.g., ``T10Y2Y``); ``timeframe`` is
    accepted but ignored — the returned series uses FRED's native
    frequency for each ID (daily/weekly/monthly). Downstream callers may
    resample as needed.
    """

    def __init__(
        self,
        api_key: str,
        *,
        client_factory: Callable[[str], _FREDClientLike] = _default_client_factory,
    ) -> None:
        self._api_key = api_key
        self._client_factory = client_factory
        self._client: _FREDClientLike | None = None

    @property
    def name(self) -> str:
        return "fred"

    @property
    def supported_asset_classes(self) -> list[str]:
        return ["macro"]

    def validate_credentials(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> _FREDClientLike:
        if self._client is None:
            self._client = self._client_factory(self._api_key)
        return self._client

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        timeframe: str = "1d",  # noqa: ARG002 — FRED uses native frequency
    ) -> pl.DataFrame:
        client = self._get_client()
        series = client.get_series(
            symbol, observation_start=start, observation_end=end
        )
        rows: list[dict[str, Any]] = []
        try:
            iter_items = list(series.items())
        except AttributeError:  # pragma: no cover - defensive
            iter_items = []
        for ts, value in iter_items:
            v = float(value)
            if v != v:  # NaN check (FRED may return NaN for missing obs)
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "open": v,
                    "high": v,
                    "low": v,
                    "close": v,
                    "volume": 0.0,
                }
            )
        if not rows:
            return pl.DataFrame(schema=_EMPTY_SCHEMA)
        df = pl.DataFrame(rows).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("open").cast(PRICE_DTYPE),
            pl.col("high").cast(PRICE_DTYPE),
            pl.col("low").cast(PRICE_DTYPE),
            pl.col("close").cast(PRICE_DTYPE),
            pl.col("volume").cast(VOLUME_DTYPE),
        )
        return df

    def list_instruments(
        self, asset_class: str | None = None  # noqa: ARG002 — see below
    ) -> pl.DataFrame:
        # FRED has 800k+ series; no useful global listing endpoint. Callers
        # query by series_id directly. Return empty schema-conforming frame.
        return pl.DataFrame(
            schema={"symbol": pl.String, "name": pl.String, "asset_class": pl.String}
        )


__all__ = ["FREDProvider"]

"""Tests for the FRED provider (Federal Reserve Economic Data)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import polars as pl
import pytest

from liq.data.providers.fred import FREDProvider
from liq.data.settings import LiqDataSettings, create_fred_provider


class FakeFredClient:
    """In-process fake of the fredapi.Fred client for unit tests.

    Holds canned series data; records calls so tests can assert args.
    """

    def __init__(self, series: dict[str, pd.Series] | None = None) -> None:
        self.series = series or {}
        self.calls: list[dict[str, Any]] = []

    def get_series(
        self,
        series_id: str,
        observation_start: date | str | None = None,
        observation_end: date | str | None = None,
    ) -> pd.Series:
        self.calls.append(
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
            }
        )
        if series_id not in self.series:
            return pd.Series(dtype=float)
        return self.series[series_id]


class TestFREDProvider:
    def test_name_and_asset_classes(self) -> None:
        p = FREDProvider(api_key="dummy", client_factory=lambda _: FakeFredClient())
        assert p.name == "fred"
        assert "macro" in p.supported_asset_classes

    def test_validate_credentials_true_when_api_key_present(self) -> None:
        p = FREDProvider(api_key="real_key", client_factory=lambda _: FakeFredClient())
        assert p.validate_credentials() is True

    def test_validate_credentials_false_when_api_key_empty(self) -> None:
        p = FREDProvider(api_key="", client_factory=lambda _: FakeFredClient())
        assert p.validate_credentials() is False

    def test_fetch_bars_returns_ohlcv_with_value_as_close_open_high_low(self) -> None:
        # FRED series is a single time-value pair per row; OHLC all = value,
        # volume = 0 (no traded-volume concept for macro series).
        series = pd.Series(
            data=[2.5, 2.7, 2.6],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        client = FakeFredClient({"T10Y2Y": series})
        p = FREDProvider(api_key="k", client_factory=lambda _: client)
        df = p.fetch_bars("T10Y2Y", start=date(2024, 1, 1), end=date(2024, 1, 31))

        assert df.height == 3
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            assert col in df.columns
        closes = [float(v) for v in df["close"]]
        opens = [float(v) for v in df["open"]]
        highs = [float(v) for v in df["high"]]
        lows = [float(v) for v in df["low"]]
        vols = [float(v) for v in df["volume"]]
        assert closes == [2.5, 2.7, 2.6]
        assert opens == closes and highs == closes and lows == closes
        assert vols == [0.0, 0.0, 0.0]

    def test_fetch_bars_passes_date_range_to_client(self) -> None:
        client = FakeFredClient({"INDPRO": pd.Series(dtype=float)})
        p = FREDProvider(api_key="k", client_factory=lambda _: client)
        p.fetch_bars("INDPRO", start=date(2020, 1, 1), end=date(2025, 6, 30))
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["series_id"] == "INDPRO"
        assert call["observation_start"] == date(2020, 1, 1)
        assert call["observation_end"] == date(2025, 6, 30)

    def test_fetch_bars_empty_series_returns_empty_dataframe_with_schema(self) -> None:
        client = FakeFredClient({"UNRATE": pd.Series(dtype=float)})
        p = FREDProvider(api_key="k", client_factory=lambda _: client)
        df = p.fetch_bars("UNRATE", start=date(2024, 1, 1), end=date(2024, 12, 31))
        assert df.is_empty()
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_fetch_bars_skips_null_observations(self) -> None:
        # FRED occasionally returns NaN for missing observations; skip them.
        import numpy as np

        series = pd.Series(
            data=[1.0, np.nan, 3.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        client = FakeFredClient({"NAPM": series})
        p = FREDProvider(api_key="k", client_factory=lambda _: client)
        df = p.fetch_bars("NAPM", start=date(2024, 1, 1), end=date(2024, 1, 31))
        assert df.height == 2
        closes = [float(v) for v in df["close"]]
        assert closes == [1.0, 3.0]

    def test_list_instruments_returns_polars_dataframe(self) -> None:
        # FRED has hundreds of thousands of series; we expose only a small
        # curated set of macro indicators by default. Implementation choice:
        # return empty DataFrame with the standard schema (callers query by
        # series_id directly).
        p = FREDProvider(api_key="k", client_factory=lambda _: FakeFredClient())
        df = p.list_instruments()
        assert isinstance(df, pl.DataFrame)

    def test_get_client_constructs_once(self) -> None:
        calls: list[str] = []

        def factory(api_key: str) -> FakeFredClient:
            calls.append(api_key)
            return FakeFredClient()

        p = FREDProvider(api_key="k", client_factory=factory)
        assert p._get_client() is p._get_client()
        assert calls == ["k"]

    def test_fetch_bars_handles_non_series_response_as_empty(self) -> None:
        class BadClient:
            def get_series(
                self,
                series_id: str,
                observation_start: date | str | None = None,
                observation_end: date | str | None = None,
            ) -> object:
                return object()

        p = FREDProvider(api_key="k", client_factory=lambda _: BadClient())
        df = p.fetch_bars("BAD", start=date(2024, 1, 1), end=date(2024, 1, 31))
        assert df.is_empty()


class TestCreateFREDProvider:
    def test_create_fred_provider_uses_settings_key(self) -> None:
        settings = LiqDataSettings(fred_api_key="fred-key")
        provider = create_fred_provider(settings)
        assert isinstance(provider, FREDProvider)
        assert provider.validate_credentials() is True

    def test_create_fred_provider_requires_api_key(self) -> None:
        settings = LiqDataSettings(fred_api_key=None)
        with pytest.raises(ValueError, match="FRED_API_KEY not configured"):
            create_fred_provider(settings)

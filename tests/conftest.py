"""Pytest fixtures for liq.data tests."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest


@pytest.fixture
def sample_timestamp() -> datetime:
    """Provide a sample timezone-aware timestamp for tests."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)


@pytest.fixture
def sample_bars_data(sample_timestamp: datetime) -> list[dict]:
    """Provide sample OHLCV bar data."""
    return [
        {
            "timestamp": sample_timestamp,
            "open": 1.0850,
            "high": 1.0875,
            "low": 1.0825,
            "close": 1.0860,
            "volume": 1000.0,
        },
        {
            "timestamp": sample_timestamp + timedelta(hours=1),
            "open": 1.0860,
            "high": 1.0890,
            "low": 1.0850,
            "close": 1.0885,
            "volume": 1500.0,
        },
        {
            "timestamp": sample_timestamp + timedelta(hours=2),
            "open": 1.0885,
            "high": 1.0900,
            "low": 1.0870,
            "close": 1.0895,
            "volume": 1200.0,
        },
    ]


@pytest.fixture
def sample_bars_df(sample_bars_data: list[dict]) -> pl.DataFrame:
    """Provide sample OHLCV DataFrame."""
    return pl.DataFrame(sample_bars_data)


@pytest.fixture
def sample_instruments_df() -> pl.DataFrame:
    """Provide sample instruments DataFrame."""
    return pl.DataFrame(
        {
            "symbol": ["EUR_USD", "GBP_USD", "USD_JPY"],
            "name": ["Euro/US Dollar", "British Pound/US Dollar", "US Dollar/Japanese Yen"],
            "asset_class": ["forex", "forex", "forex"],
            "base_currency": ["EUR", "GBP", "USD"],
            "quote_currency": ["USD", "USD", "JPY"],
        }
    )

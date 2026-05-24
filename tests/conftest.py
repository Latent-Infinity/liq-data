"""Pytest fixtures for liq.data tests."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest


def _noop_persist_env_value(*_args: object, **_kwargs: object) -> None:
    return None


@pytest.fixture(autouse=True)
def _block_real_env_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent any test from writing the developer's real .env file.

    A provider that mocks the OAuth refresh endpoint will receive a fake
    rotated refresh token and, under the default
    ``tradestation_persist_refresh_token=True`` setting, will call
    ``persist_env_value`` to write it. That's destructive in a dev shell
    where the real ``.env`` is on disk - the user has already lost a
    working refresh token to this exact path. Hard-block at the
    test-isolation layer.
    """
    import liq.data.settings as _settings

    monkeypatch.setattr(_settings, "persist_env_value", _noop_persist_env_value)


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

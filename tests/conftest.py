"""Pytest fixtures for liq.data tests."""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest


def _noop_persist_env_value(*_args: object, **_kwargs: object) -> None:
    return None


@pytest.fixture(autouse=True)
def _hermetic_data_root(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Point the default data root at a throwaway directory for every test.

    ``DATA_ROOT`` in the developer's ``.env`` names the real store, so a test
    that constructs ``DataService()`` without an explicit ``data_root`` reads
    and writes there. Its declared reads then append to the production lockbox
    usage ledger, which makes that ledger's entry counts, date ranges and arm
    coverage a function of how often somebody ran the suite rather than of
    research activity. The ledger is append-only by design and cannot be
    cleaned up afterwards, so the isolation has to happen before the write.

    An environment variable is used rather than a patch of ``LiqDataSettings``
    because it is the same channel production configuration travels through, so
    nothing in the settings chain is bypassed by the test harness. The cached
    settings and store are cleared on both sides of the test so neither a
    previously cached real root leaks in nor a temporary root leaks out.

    Tests marked ``realdata`` deliberately resolve against the real store and
    are exempt; they are opt-in behind ``RUN_REALDATA=1``.
    """
    import liq.data.settings as _settings

    if request.node.get_closest_marker("realdata") is not None:
        yield
        return

    monkeypatch.setenv("DATA_ROOT", str(tmp_path_factory.mktemp("hermetic_data_root")))
    _settings.get_settings.cache_clear()
    _settings.get_store.cache_clear()
    try:
        yield
    finally:
        _settings.get_settings.cache_clear()
        _settings.get_store.cache_clear()


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

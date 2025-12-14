"""Tests for liq.data.settings module."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from liq.data.settings import (
    LiqDataSettings,
    create_alpaca_provider,
    create_binance_provider,
    create_coinbase_provider,
    create_oanda_provider,
    create_polygon_provider,
    create_tradestation_provider,
    get_settings,
    get_storage_key,
    get_store,
    list_available_data,
    load_symbol_data,
)
from liq.store.parquet import ParquetStore


class TestLiqDataSettings:
    """Tests for LiqDataSettings class."""

    def test_default_values(self) -> None:
        """Test default settings values."""
        # Clear any cached settings
        get_settings.cache_clear()

        with patch.dict(os.environ, {}, clear=True):
            # Pass _env_file=None to prevent loading .env file
            settings = LiqDataSettings(_env_file=None)

        assert settings.oanda_api_key is None
        assert settings.oanda_account_id is None
        assert settings.oanda_environment == "practice"
        assert settings.binance_api_key is None
        assert settings.binance_use_us is False
        # data_root is resolved to absolute path
        assert settings.data_root.is_absolute()
        assert settings.data_root.name == "financial_data"
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"
        assert settings.log_file is None

    def test_from_environment(self) -> None:
        """Test settings loaded from environment variables."""
        env = {
            "OANDA_API_KEY": "test_key",
            "OANDA_ACCOUNT_ID": "test_account",
            "OANDA_ENVIRONMENT": "live",
            "BINANCE_USE_US": "true",
            "DATA_ROOT": "/custom/path",
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "console",
            "LOG_FILE": "/var/log/app.log",
        }

        with patch.dict(os.environ, env, clear=True):
            # Pass _env_file=None to ensure only env vars are used
            settings = LiqDataSettings(_env_file=None)

        assert settings.oanda_api_key == "test_key"
        assert settings.oanda_account_id == "test_account"
        assert settings.oanda_environment == "live"
        assert settings.binance_use_us is True
        # Absolute path stays absolute after resolution
        assert settings.data_root == Path("/custom/path")
        assert settings.data_root.is_absolute()
        assert settings.log_level == "DEBUG"
        assert settings.log_format == "console"
        assert settings.log_file == Path("/var/log/app.log")


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings(self) -> None:
        """Test get_settings returns LiqDataSettings instance."""
        get_settings.cache_clear()

        settings = get_settings()

        assert isinstance(settings, LiqDataSettings)

    def test_caches_result(self) -> None:
        """Test get_settings caches the result."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2


class TestCreateOandaProvider:
    """Tests for create_oanda_provider function."""

    def test_missing_api_key_raises(self) -> None:
        """Test missing API key raises ValueError."""
        settings = LiqDataSettings(oanda_api_key=None, oanda_account_id="account")

        with pytest.raises(ValueError, match="OANDA_API_KEY not configured"):
            create_oanda_provider(settings)

    def test_missing_account_id_raises(self) -> None:
        """Test missing account ID raises ValueError."""
        settings = LiqDataSettings(oanda_api_key="key", oanda_account_id=None)

        with pytest.raises(ValueError, match="OANDA_ACCOUNT_ID not configured"):
            create_oanda_provider(settings)

    def test_creates_provider_with_settings(self) -> None:
        """Test creates provider with correct settings."""
        settings = LiqDataSettings(
            oanda_api_key="test_key",
            oanda_account_id="test_account",
            oanda_environment="practice",
        )

        provider = create_oanda_provider(settings)

        assert provider.name == "oanda"
        assert provider._api_key == "test_key"
        assert provider._account_id == "test_account"
        assert provider._environment == "practice"


class TestCreateBinanceProvider:
    """Tests for create_binance_provider function."""

    def test_creates_provider_without_auth(self) -> None:
        """Test creates provider without authentication."""
        with patch.dict(os.environ, {}, clear=True):
            settings = LiqDataSettings(_env_file=None)

        provider = create_binance_provider(settings)

        assert provider.name == "binance"
        assert provider._api_key is None

    def test_creates_provider_with_auth(self) -> None:
        """Test creates provider with authentication."""
        settings = LiqDataSettings(
            binance_api_key="test_key",
            binance_api_secret="test_secret",
            binance_use_us=True,
        )

        provider = create_binance_provider(settings)

        assert provider.name == "binance_us"
        assert provider._api_key == "test_key"
        assert provider._api_secret == "test_secret"
        assert provider._use_us is True

    def test_uses_default_settings_if_none(self) -> None:
        """Test uses get_settings() if settings not provided."""
        get_settings.cache_clear()

        # Should not raise - uses default settings
        provider = create_binance_provider()

        # Provider name depends on BINANCE_USE_US env var setting
        assert provider.name in ("binance", "binance_us")


class TestCreatePolygonProvider:
    """Tests for create_polygon_provider function."""

    def test_missing_api_key_raises(self) -> None:
        """Test missing API key raises ValueError."""
        settings = LiqDataSettings(polygon_api_key=None)

        with pytest.raises(ValueError, match="POLYGON_API_KEY not configured"):
            create_polygon_provider(settings)

    def test_creates_provider_with_settings(self) -> None:
        """Test creates provider with correct settings."""
        settings = LiqDataSettings(polygon_api_key="test_key")

        provider = create_polygon_provider(settings)

        assert provider.name == "polygon"
        assert provider._api_key == "test_key"


class TestCreateTradeStationProvider:
    """Tests for create_tradestation_provider function."""

    def test_missing_client_id_raises(self) -> None:
        """Test missing client ID raises ValueError."""
        settings = LiqDataSettings(
            tradestation_client_id=None,
            tradestation_client_secret="secret",
            tradestation_refresh_token="token",
        )

        with pytest.raises(ValueError, match="TRADESTATION_CLIENT_ID not configured"):
            create_tradestation_provider(settings)

    def test_missing_client_secret_raises(self) -> None:
        """Test missing client secret raises ValueError."""
        settings = LiqDataSettings(
            tradestation_client_id="client_id",
            tradestation_client_secret=None,
            tradestation_refresh_token="token",
        )

        with pytest.raises(ValueError, match="TRADESTATION_CLIENT_SECRET not configured"):
            create_tradestation_provider(settings)

    def test_missing_refresh_token_raises(self) -> None:
        """Test missing refresh token raises ValueError."""
        settings = LiqDataSettings(
            tradestation_client_id="client_id",
            tradestation_client_secret="secret",
            tradestation_refresh_token=None,
        )

        with pytest.raises(ValueError, match="TRADESTATION_REFRESH_TOKEN not configured"):
            create_tradestation_provider(settings)

    def test_creates_provider_with_settings(self) -> None:
        """Test creates provider with correct settings."""
        settings = LiqDataSettings(
            tradestation_client_id="test_client_id",
            tradestation_client_secret="test_secret",
            tradestation_refresh_token="test_token",
        )

        provider = create_tradestation_provider(settings)

        assert provider.name == "tradestation"
        assert provider._client_id == "test_client_id"
        assert provider._client_secret == "test_secret"


class TestCreateCoinbaseProvider:
    """Tests for create_coinbase_provider function."""

    def test_missing_api_key_raises(self) -> None:
        """Test missing API key raises ValueError."""
        settings = LiqDataSettings(
            coinbase_api_key=None,
            coinbase_api_secret="secret",
            coinbase_passphrase="pass",
        )

        with pytest.raises(ValueError, match="COINBASE_API_KEY not configured"):
            create_coinbase_provider(settings)

    def test_missing_api_secret_raises(self) -> None:
        """Test missing API secret raises ValueError."""
        settings = LiqDataSettings(
            coinbase_api_key="key",
            coinbase_api_secret=None,
            coinbase_passphrase="pass",
        )

        with pytest.raises(ValueError, match="COINBASE_API_SECRET not configured"):
            create_coinbase_provider(settings)

    def test_missing_passphrase_raises(self) -> None:
        """Test missing passphrase raises ValueError."""
        settings = LiqDataSettings(
            coinbase_api_key="key",
            coinbase_api_secret="secret",
            coinbase_passphrase=None,
        )

        with pytest.raises(ValueError, match="COINBASE_PASSPHRASE not configured"):
            create_coinbase_provider(settings)

    def test_creates_provider_with_settings(self) -> None:
        """Test creates provider with correct settings."""
        # Use valid base64-encoded secret for HMAC signing
        settings = LiqDataSettings(
            coinbase_api_key="test_key",
            coinbase_api_secret="dGVzdF9zZWNyZXQ=",  # base64("test_secret")
            coinbase_passphrase="test_passphrase",
        )

        provider = create_coinbase_provider(settings)

        assert provider.name == "coinbase"
        assert provider._api_key == "test_key"
        assert provider._passphrase == "test_passphrase"


class TestCreateAlpacaProvider:
    """Tests for create_alpaca_provider function."""

    def test_missing_api_key_raises(self) -> None:
        """Test missing API key raises ValueError."""
        settings = LiqDataSettings(
            alpaca_api_key=None,
            alpaca_api_secret="secret",
        )

        with pytest.raises(ValueError, match="ALPACA_API_KEY not configured"):
            create_alpaca_provider(settings)

    def test_missing_api_secret_raises(self) -> None:
        """Test missing API secret raises ValueError."""
        settings = LiqDataSettings(
            alpaca_api_key="key",
            alpaca_api_secret=None,
        )

        with pytest.raises(ValueError, match="ALPACA_API_SECRET not configured"):
            create_alpaca_provider(settings)

    def test_creates_provider_with_settings(self) -> None:
        """Test creates provider with correct settings."""
        settings = LiqDataSettings(
            alpaca_api_key="test_key",
            alpaca_api_secret="test_secret",
        )

        provider = create_alpaca_provider(settings)

        assert provider.name == "alpaca"
        assert provider._api_key == "test_key"
        assert provider._api_secret == "test_secret"


class TestGetStorageKey:
    """Tests for get_storage_key function."""

    def test_returns_correct_key(self) -> None:
        """Test get_storage_key returns correct key structure."""
        key = get_storage_key("oanda", "EUR_USD", "1m")
        assert key == "oanda/EUR_USD/bars/1m"

    def test_handles_different_providers(self) -> None:
        """Test get_storage_key works with different providers."""
        for provider in ["oanda", "binance", "polygon", "alpaca"]:
            key = get_storage_key(provider, "BTC_USDT", "1h")
            assert key == f"{provider}/BTC_USDT/bars/1h"

    def test_handles_different_timeframes(self) -> None:
        """Test get_storage_key handles various timeframes."""
        for tf in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            key = get_storage_key("oanda", "EUR_USD", tf)
            assert key == f"oanda/EUR_USD/bars/{tf}"


class TestLoadSymbolData:
    """Tests for load_symbol_data function."""

    def test_loads_existing_data(self, tmp_path: Path) -> None:
        """Test load_symbol_data loads existing data via liq-store."""
        # Create test data using ParquetStore
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        # Clear cache and mock settings
        get_settings.cache_clear()
        get_store.cache_clear()

        with patch("liq.data.settings.get_settings") as mock_settings:
            mock = LiqDataSettings(data_root=tmp_path, _env_file=None)
            mock_settings.return_value = mock

            # Also need to clear and recreate the store with the new path
            with patch("liq.data.settings.get_store") as mock_store:
                mock_store.return_value = store

                result = load_symbol_data("oanda", "EUR_USD", "1m")

                assert len(result) == 2
                assert "timestamp" in result.columns
                assert "open" in result.columns

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """Test load_symbol_data raises FileNotFoundError if data missing."""
        store = ParquetStore(str(tmp_path))

        get_settings.cache_clear()
        get_store.cache_clear()

        with patch("liq.data.settings.get_settings") as mock_settings:
            mock = LiqDataSettings(data_root=tmp_path, _env_file=None)
            mock_settings.return_value = mock

            with patch("liq.data.settings.get_store") as mock_store:
                mock_store.return_value = store

                with pytest.raises(FileNotFoundError, match="Data not found"):
                    load_symbol_data("oanda", "EUR_USD", "1m")

    def test_error_message_contains_fetch_hint(self, tmp_path: Path) -> None:
        """Test error message contains hint to use fetch command."""
        store = ParquetStore(str(tmp_path))

        get_settings.cache_clear()
        get_store.cache_clear()

        with patch("liq.data.settings.get_settings") as mock_settings:
            mock = LiqDataSettings(data_root=tmp_path, _env_file=None)
            mock_settings.return_value = mock

            with patch("liq.data.settings.get_store") as mock_store:
                mock_store.return_value = store

                with pytest.raises(FileNotFoundError, match="liq-data fetch"):
                    load_symbol_data("binance", "BTC_USDT", "1h")


class TestListAvailableData:
    """Tests for list_available_data function."""

    def test_returns_empty_list_if_no_data(self, tmp_path: Path) -> None:
        """Test returns empty list if no data exists in store."""
        store = ParquetStore(str(tmp_path))

        get_settings.cache_clear()
        get_store.cache_clear()

        with patch("liq.data.settings.get_settings") as mock_settings:
            mock = LiqDataSettings(data_root=tmp_path, _env_file=None)
            mock_settings.return_value = mock

            with patch("liq.data.settings.get_store") as mock_store:
                mock_store.return_value = store

                result = list_available_data()

                assert result == []

    def test_lists_available_data_files(self, tmp_path: Path) -> None:
        """Test lists all available data via liq-store."""
        # Create test data using ParquetStore
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "open": [1.0]})
        store.write("oanda/EUR_USD/bars/1m", df)
        store.write("oanda/EUR_USD/1h", df)
        store.write("binance/BTC_USDT/1d", df)

        get_settings.cache_clear()
        get_store.cache_clear()

        with patch("liq.data.settings.get_settings") as mock_settings:
            mock = LiqDataSettings(data_root=tmp_path, _env_file=None)
            mock_settings.return_value = mock

            with patch("liq.data.settings.get_store") as mock_store:
                mock_store.return_value = store

                result = list_available_data()

                assert len(result) == 3
                # Check structure of returned data
                providers = {r["provider"] for r in result}
                assert "oanda" in providers
                assert "binance" in providers

                symbols = {r["symbol"] for r in result}
                assert "EUR_USD" in symbols
                assert "BTC_USDT" in symbols

                timeframes = {r["timeframe"] for r in result}
                assert "1m" in timeframes
                assert "1h" in timeframes
                assert "1d" in timeframes

    def test_ignores_keys_with_wrong_structure(self, tmp_path: Path) -> None:
        """Test ignores storage keys not in expected format."""
        store = ParquetStore(str(tmp_path))

        # Create valid data
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 1)], "open": [1.0]})
        store.write("oanda/EUR_USD/bars/1m", df)

        get_settings.cache_clear()
        get_store.cache_clear()

        with patch("liq.data.settings.get_settings") as mock_settings:
            mock = LiqDataSettings(data_root=tmp_path, _env_file=None)
            mock_settings.return_value = mock

            with patch("liq.data.settings.get_store") as mock_store:
                mock_store.return_value = store

                result = list_available_data()

                # Should only include properly structured keys
                assert len(result) == 1
                assert all(len(r) == 3 for r in result)  # provider, symbol, timeframe

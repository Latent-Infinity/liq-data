"""Tests for liq.data.cli module."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from typer.testing import CliRunner
from liq.store import key_builder

from liq.data.cli import app
from liq.data.cli.common import create_fetch_progress, get_provider, parse_date, parse_source_spec, storage_key
from liq.data.exceptions import AuthenticationError, ProviderError
import typer
from liq.store.parquet import ParquetStore

runner = CliRunner()


def write_test_data(
    tmp_path: Path,
    provider: str,
    symbol: str,
    timeframe: str,
    df: pl.DataFrame,
) -> None:
    """Write test data using ParquetStore."""
    store = ParquetStore(str(tmp_path))
    storage_key = f"{provider}/{key_builder.bars(symbol, timeframe)}"
    store.write(storage_key, df)


class TestParseDate:
    """Tests for parse_date helper function."""

    def test_parses_valid_date(self) -> None:
        """Test parsing a valid date string."""
        result = parse_date("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_parses_boundary_dates(self) -> None:
        """Test parsing boundary dates."""
        assert parse_date("2000-01-01") == date(2000, 1, 1)
        assert parse_date("2024-12-31") == date(2024, 12, 31)

    def test_invalid_format_raises(self) -> None:
        """Test that invalid date format raises ValueError."""
        with pytest.raises(ValueError):
            parse_date("01-15-2024")

    def test_invalid_date_raises(self) -> None:
        """Test that invalid date raises ValueError."""
        with pytest.raises(ValueError):
            parse_date("2024-02-30")


class TestCommonHelpers:
    """Tests for CLI common helpers."""

    def test_storage_key_uses_bars_layout(self) -> None:
        assert storage_key("oanda", "EUR_USD", "1m") == "oanda/EUR_USD/bars/1m"

    def test_parse_source_spec_invalid(self) -> None:
        with pytest.raises(ValueError):
            parse_source_spec("oanda")

    def test_get_provider_unknown(self) -> None:
        with pytest.raises(typer.Exit):
            get_provider("unknown")

    def test_get_provider_config_error(self) -> None:
        with patch("liq.data.cli.common.create_oanda_provider", side_effect=ValueError("bad config")):
            with pytest.raises(typer.Exit):
                get_provider("oanda")

    def test_get_provider_supported_names(self) -> None:
        with patch("liq.data.cli.common.create_binance_provider", return_value=MagicMock()):
            assert get_provider("binance")
        with patch("liq.data.cli.common.create_tradestation_provider", return_value=MagicMock()):
            assert get_provider("tradestation")
        with patch("liq.data.cli.common.create_coinbase_provider", return_value=MagicMock()):
            assert get_provider("coinbase")
        with patch("liq.data.cli.common.create_polygon_provider", return_value=MagicMock()):
            assert get_provider("polygon")
        with patch("liq.data.cli.common.create_alpaca_provider", return_value=MagicMock()):
            assert get_provider("alpaca")

    def test_create_fetch_progress(self) -> None:
        progress = create_fetch_progress()
        assert progress is not None


class TestConfigCommand:
    """Tests for the config command."""

    def test_shows_config_table(self) -> None:
        """Test config command shows configuration table."""
        result = runner.invoke(app, ["config"])

        assert result.exit_code == 0
        assert "LIQ Data Configuration" in result.output
        assert "OANDA" in result.output
        assert "Binance" in result.output
        assert "Data Root" in result.output
        assert "Log Level" in result.output

    def test_masks_api_keys(self) -> None:
        """Test config command masks API keys."""
        with patch("liq.data.cli.info.get_settings") as mock_settings:
            mock = MagicMock()
            mock.oanda_api_key = "secret_key"
            mock.oanda_account_id = "account123"
            mock.oanda_environment = "practice"
            mock.binance_api_key = "binance_key"
            mock.binance_use_us = False
            mock.data_root = Path("/data")
            mock.log_level = "INFO"
            mock.log_format = "json"
            mock.log_file = None
            mock_settings.return_value = mock

            result = runner.invoke(app, ["config"])

            assert result.exit_code == 0
            assert "***" in result.output
            assert "secret_key" not in result.output


class TestInfoCommand:
    """Tests for the info command."""

    def test_no_data_shows_warning(self, tmp_path: Path) -> None:
        """Test info command when no data exists."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.info.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert "No data available" in result.output

    def test_empty_store_shows_message(self, tmp_path: Path) -> None:
        """Test info command when store is empty."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.info.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert "No data available" in result.output

    def test_lists_available_data(self, tmp_path: Path) -> None:
        """Test info command lists available data via liq-store."""
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

        with patch("liq.data.cli.info.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert "Available Data" in result.output
            assert "oanda" in result.output
            assert "EUR_USD" in result.output

    def test_specific_symbol_not_found(self, tmp_path: Path) -> None:
        """Test info command for specific symbol that doesn't exist."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.info.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["info", "oanda", "EUR_USD"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_specific_symbol_shows_details(self, tmp_path: Path) -> None:
        """Test info command shows details for specific symbol."""
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

        with patch("liq.data.cli.info.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["info", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            assert "Data Summary" in result.output
            assert "Total Bars" in result.output


class TestValidateCommand:
    """Tests for the validate command."""

    def test_data_not_found(self, tmp_path: Path) -> None:
        """Test validate command when data doesn't exist."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_validates_clean_data(self, tmp_path: Path) -> None:
        """Test validate command with clean data."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 1),
                datetime(2024, 1, 1, 10, 2),
            ],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [100.0, 200.0, 150.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            assert "Validation Summary" in result.output


class TestAuthCommands:
    """Tests for TradeStation auth helpers."""

    def test_tradestation_auth_url_requires_client_id(self) -> None:
        """Missing client ID should return an error."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = None
            mock.tradestation_redirect_uri = "https://example.com/callback"
            mock.tradestation_scopes = "scope"
            mock_settings.return_value = mock

            result = runner.invoke(app, ["tradestation-auth-url"])

            assert result.exit_code == 1
            assert "not configured" in result.output.lower()

    def test_tradestation_auth_url_success(self) -> None:
        """Auth URL command should print state and URL."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = "client"
            mock.tradestation_redirect_uri = "https://example.com/callback"
            mock.tradestation_scopes = "scope"
            mock_settings.return_value = mock

            with patch("liq.data.providers.tradestation.TradeStationProvider.build_authorization_url") as mock_build:
                mock_build.return_value = "https://auth.example.com"
                result = runner.invoke(app, ["tradestation-auth-url", "--state", "state123"])

            assert result.exit_code == 0
            assert "state123" in result.output
            assert "https://auth.example.com" in result.output

    def test_tradestation_auth_url_requires_redirect(self) -> None:
        """Missing redirect should return an error."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = "client"
            mock.tradestation_redirect_uri = None
            mock.tradestation_scopes = "scope"
            mock_settings.return_value = mock

            result = runner.invoke(app, ["tradestation-auth-url"])

            assert result.exit_code == 1
            assert "redirect uri" in result.output.lower()

    def test_tradestation_exchange_code_requires_secret(self) -> None:
        """Missing client secret should return an error."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = "client"
            mock.tradestation_client_secret = None
            mock.tradestation_redirect_uri = "https://example.com/callback"
            mock_settings.return_value = mock

            result = runner.invoke(app, ["tradestation-exchange-code", "code123"])

            assert result.exit_code == 1
            assert "not configured" in result.output.lower()

    def test_tradestation_exchange_code_success(self) -> None:
        """Exchange code command should print refresh token."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = "client"
            mock.tradestation_client_secret = "secret"
            mock.tradestation_redirect_uri = "https://example.com/callback"
            mock_settings.return_value = mock

            with patch("liq.data.providers.tradestation.TradeStationProvider.exchange_authorization_code") as mock_exchange:
                mock_exchange.return_value = {"refresh_token": "refresh123"}
                result = runner.invoke(app, ["tradestation-exchange-code", "code123"])

            assert result.exit_code == 0
            assert "refresh123" in result.output

    def test_tradestation_exchange_code_auth_failure(self) -> None:
        """Auth exchange errors should be surfaced."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = "client"
            mock.tradestation_client_secret = "secret"
            mock.tradestation_redirect_uri = "https://example.com/callback"
            mock_settings.return_value = mock

            with patch("liq.data.providers.tradestation.TradeStationProvider.exchange_authorization_code") as mock_exchange:
                mock_exchange.side_effect = AuthenticationError("bad code")
                result = runner.invoke(app, ["tradestation-exchange-code", "code123"])

            assert result.exit_code == 1
            assert "auth code exchange failed" in result.output.lower()

    def test_tradestation_exchange_code_requires_redirect(self) -> None:
        """Missing redirect should return an error."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = "client"
            mock.tradestation_client_secret = "secret"
            mock.tradestation_redirect_uri = None
            mock_settings.return_value = mock

            result = runner.invoke(app, ["tradestation-exchange-code", "code123"])

            assert result.exit_code == 1
            assert "redirect" in result.output.lower()

    def test_tradestation_exchange_code_missing_refresh_token(self) -> None:
        """Missing refresh token should be surfaced."""
        with patch("liq.data.cli.auth.get_settings") as mock_settings:
            mock = MagicMock()
            mock.tradestation_client_id = "client"
            mock.tradestation_client_secret = "secret"
            mock.tradestation_redirect_uri = "https://example.com/callback"
            mock_settings.return_value = mock

            with patch("liq.data.providers.tradestation.TradeStationProvider.exchange_authorization_code") as mock_exchange:
                mock_exchange.return_value = {"access_token": "token"}
                result = runner.invoke(app, ["tradestation-exchange-code", "code123"])

            assert result.exit_code == 1
            assert "no refresh token" in result.output.lower()


class TestFetchCommandEdgeCases:
    """Tests for fetch command error handling and validate edge cases."""

    def test_fetch_handles_value_error(self) -> None:
        with patch("liq.data.cli.fetch.DataService") as mock_service:
            mock_service.return_value.fetch.side_effect = ValueError("bad input")
            result = runner.invoke(
                app,
                ["fetch", "oanda", "EUR_USD", "--start", "2024-01-01"],
            )
        assert result.exit_code == 1
        assert "bad input" in result.output.lower()

    def test_fetch_handles_provider_error(self) -> None:
        with patch("liq.data.cli.fetch.DataService") as mock_service:
            mock_service.return_value.fetch.side_effect = ProviderError("provider down")
            result = runner.invoke(
                app,
                ["fetch", "oanda", "EUR_USD", "--start", "2024-01-01"],
            )
        assert result.exit_code == 1
        assert "provider error" in result.output.lower()

    def test_fetch_no_data_returns_success(self) -> None:
        mock_df = pl.DataFrame({"timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC"))})
        with patch("liq.data.cli.fetch.DataService") as mock_service:
            mock_service.return_value.fetch.return_value = mock_df
            result = runner.invoke(
                app,
                ["fetch", "oanda", "EUR_USD", "--start", "2024-01-01"],
            )
        assert result.exit_code == 0
        assert "no data returned" in result.output.lower()

    def test_fetch_dry_run_shows_preview(self) -> None:
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 200.0],
        })
        with patch("liq.data.cli.fetch.DataService") as mock_service:
            mock_service.return_value.fetch.return_value = df
            result = runner.invoke(
                app,
                ["fetch", "oanda", "EUR_USD", "--start", "2024-01-01", "--dry-run"],
            )
        assert result.exit_code == 0
        assert "Fetched" in result.output
        assert "Stored via liq-store" not in result.output

    def test_backfill_runs_successfully(self) -> None:
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        with patch("liq.data.cli.fetch.DataService") as mock_service:
            mock_service.return_value.backfill.return_value = df
            result = runner.invoke(
                app,
                ["backfill", "oanda", "EUR_USD", "--start", "2024-01-01", "--end", "2024-01-02"],
            )
        assert result.exit_code == 0
        assert "Backfill complete" in result.output

    def test_detects_null_values(self, tmp_path: Path) -> None:
        """Test validate command detects null values."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "open": [1.0, None],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            assert "Null values" in result.output

    def test_no_duplicates_after_liq_store_write(self, tmp_path: Path) -> None:
        """Test that liq-store deduplicates data on write, so validate passes."""
        store = ParquetStore(str(tmp_path))

        # Write data with duplicate timestamps - liq-store will deduplicate
        ts = datetime(2024, 1, 1)
        df = pl.DataFrame({
            "timestamp": [ts, ts],  # Duplicate - will be deduplicated by liq-store
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        # Verify liq-store deduplicated the data
        stored_df = store.read("oanda/EUR_USD/bars/1m")
        assert stored_df.n_unique(subset=["timestamp"]) == len(stored_df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            # No duplicate warning since liq-store deduplicated on write
            assert "Duplicate" not in result.output

    def test_detects_ohlc_inconsistency(self, tmp_path: Path) -> None:
        """Test validate command detects OHLC inconsistency (high < low)."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [0.8],  # Invalid: high < low
            "low": [1.2],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            assert "High < Low" in result.output

    def test_detects_negative_values(self, tmp_path: Path) -> None:
        """Test validate command detects negative values."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [-1.0],  # Negative
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            assert "Negative" in result.output

    def test_detects_duplicates_unsorted_and_gaps(self) -> None:
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 2),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 10),
            ],
            "open": [1.0, 1.0, 1.0, 1.1],
            "high": [1.1, 1.1, 1.1, 1.2],
            "low": [0.9, 0.9, 0.9, 1.0],
            "close": [1.05, 1.05, 1.05, 1.15],
            "volume": [100.0, 100.0, 100.0, 200.0],
        })
        store = MagicMock()
        store.exists.return_value = True
        store.read.return_value = df
        with patch("liq.data.cli.validate.get_store", return_value=store):
            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

        assert result.exit_code == 0
        assert "Duplicate timestamps" in result.output
        assert "Timestamps not sorted" in result.output
        assert "Unexpected gaps" in result.output

    def test_detects_multiple_ohlc_consistency_issues(self) -> None:
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 0, 0),
                datetime(2024, 1, 1, 0, 1),
                datetime(2024, 1, 1, 0, 2),
                datetime(2024, 1, 1, 0, 3),
            ],
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [0.9, 0.9, 1.2, 1.2],
            "low": [0.8, 0.7, 1.1, 0.8],
            "close": [0.85, 1.1, 1.05, 0.7],
            "volume": [100.0, 100.0, 100.0, 100.0],
        })
        store = MagicMock()
        store.exists.return_value = True
        store.read.return_value = df
        with patch("liq.data.cli.validate.get_store", return_value=store):
            result = runner.invoke(app, ["validate", "oanda", "EUR_USD"])

        assert result.exit_code == 0
        assert "High < Open" in result.output
        assert "High < Close" in result.output
        assert "Low > Open" in result.output
        assert "Low > Close" in result.output


class TestStatsCommand:
    """Tests for the stats command."""

    def test_data_not_found(self, tmp_path: Path) -> None:
        """Test stats command when data doesn't exist."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.info.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["stats", "oanda", "EUR_USD"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_shows_statistics(self, tmp_path: Path) -> None:
        """Test stats command shows OHLCV statistics."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [100.0, 200.0, 150.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.info.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["stats", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            assert "OHLCV Statistics" in result.output
            assert "Time Coverage" in result.output
            assert "Yearly Breakdown" in result.output


class TestFetchCommand:
    """Tests for the fetch command."""

    def test_unknown_provider_fails(self) -> None:
        """Test fetch with unknown provider fails."""
        result = runner.invoke(app, ["fetch", "unknown", "EUR_USD", "--start", "2024-01-01"])

        assert result.exit_code == 1
        assert "Unknown provider" in result.output

    def test_missing_api_key_fails(self, tmp_path: Path) -> None:
        """Test fetch with missing API key fails."""
        with (
            patch("liq.data.service.get_settings") as mock_settings,
            patch.dict(
                "liq.data.service.DataService._PROVIDER_FACTORIES",
                {"oanda": lambda _settings: (_ for _ in ()).throw(ValueError("OANDA_API_KEY not configured"))},
            ),
        ):
            settings = MagicMock()
            settings.data_root = tmp_path
            mock_settings.return_value = settings

            result = runner.invoke(app, ["fetch", "oanda", "EUR_USD", "--start", "2024-01-01"])

            assert result.exit_code == 1
            assert "OANDA_API_KEY" in result.output or "Invalid" in result.output

    def test_fetch_error_shows_message(self, tmp_path: Path) -> None:
        """Test fetch command shows error message on fetch failure."""
        mock_provider = MagicMock()
        mock_provider.fetch_bars.side_effect = ProviderError("API error")

        with (
            patch("liq.data.service.get_settings") as mock_settings,
            patch.dict(
                "liq.data.service.DataService._PROVIDER_FACTORIES",
                {"oanda": lambda _settings: mock_provider},
            ),
        ):
            settings = MagicMock()
            settings.data_root = tmp_path
            mock_settings.return_value = settings

            result = runner.invoke(app, ["fetch", "oanda", "EUR_USD", "--start", "2024-01-01"])

            assert result.exit_code == 1
            assert "Provider error" in result.output

    def test_successful_fetch(self, tmp_path: Path) -> None:
        """Test successful fetch command."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
            ],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 200.0],
        })
        mock_provider.fetch_bars.return_value = mock_df

        with (
            patch("liq.data.service.get_settings") as mock_settings,
            patch.dict(
                "liq.data.service.DataService._PROVIDER_FACTORIES",
                {"oanda": lambda _settings: mock_provider},
            ),
        ):
            settings = MagicMock()
            settings.data_root = tmp_path
            mock_settings.return_value = settings

            result = runner.invoke(
                app, ["fetch", "oanda", "EUR_USD", "--start", "2024-01-01", "--end", "2024-01-02"]
            )

            assert result.exit_code == 0
            assert "Fetched" in result.output
            assert "Stored via liq-store" in result.output

    def test_fetch_binance_provider(self, tmp_path: Path) -> None:
        """Test fetch with binance provider."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "open": [42000.0],
            "high": [42500.0],
            "low": [41500.0],
            "close": [42200.0],
            "volume": [100.0],
        })
        mock_provider.fetch_bars.return_value = mock_df

        with (
            patch("liq.data.service.get_settings") as mock_settings,
            patch.dict(
                "liq.data.service.DataService._PROVIDER_FACTORIES",
                {"binance": lambda _settings: mock_provider},
            ),
        ):
            settings = MagicMock()
            settings.data_root = tmp_path
            mock_settings.return_value = settings

            result = runner.invoke(
                app, ["fetch", "binance", "BTC_USDT", "--start", "2024-01-01"]
            )

            assert result.exit_code == 0
            assert "Fetched" in result.output


class TestListCommand:
    """Tests for the list command."""

    def test_unknown_provider_fails(self) -> None:
        """Test list with unknown provider fails."""
        result = runner.invoke(app, ["list", "unknown"])

        assert result.exit_code == 1
        assert "Unknown provider" in result.output

    def test_missing_api_key_fails(self) -> None:
        """Test list with missing API key fails."""
        with patch("liq.data.cli.common.create_oanda_provider") as mock_create:
            mock_create.side_effect = ValueError("OANDA_API_KEY not configured")

            result = runner.invoke(app, ["list", "oanda"])

            assert result.exit_code == 1
            assert "Configuration error" in result.output

    def test_list_error_shows_message(self) -> None:
        """Test list command shows error message on failure."""
        mock_provider = MagicMock()
        mock_provider.list_instruments.side_effect = Exception("API error")

        with patch("liq.data.cli.common.create_oanda_provider", return_value=mock_provider):
            result = runner.invoke(app, ["list", "oanda"])

            assert result.exit_code == 1
            assert "Error" in result.output

    def test_successful_list(self) -> None:
        """Test successful list command."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "symbol": ["EUR_USD", "GBP_USD"],
            "name": ["Euro/USD", "Pound/USD"],
            "asset_class": ["forex", "forex"],
            "type": ["currency", "currency"],
        })
        mock_provider.list_instruments.return_value = mock_df

        with patch("liq.data.cli.common.create_oanda_provider", return_value=mock_provider):
            result = runner.invoke(app, ["list", "oanda"])

            assert result.exit_code == 0
            assert "OANDA" in result.output
            assert "EUR_USD" in result.output

    def test_list_binance_provider(self) -> None:
        """Test list with binance provider."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "symbol": ["BTC_USDT"],
            "name": ["Bitcoin/USDT"],
            "asset_class": ["crypto"],
            "type": ["spot"],
        })
        mock_provider.list_instruments.return_value = mock_df

        with patch("liq.data.cli.common.create_binance_provider", return_value=mock_provider):
            result = runner.invoke(app, ["list", "binance"])

            assert result.exit_code == 0
            assert "BINANCE" in result.output
            assert "BTC_USDT" in result.output

    def test_list_with_asset_class_filter(self) -> None:
        """Test list command with asset class filter."""
        mock_provider = MagicMock()
        mock_df = pl.DataFrame({
            "symbol": ["EUR_USD"],
            "name": ["Euro/USD"],
            "asset_class": ["forex"],
            "type": ["currency"],
        })
        mock_provider.list_instruments.return_value = mock_df

        with patch("liq.data.cli.common.create_oanda_provider", return_value=mock_provider):
            result = runner.invoke(app, ["list", "oanda", "--asset-class", "forex"])

            assert result.exit_code == 0
            mock_provider.list_instruments.assert_called_once_with("forex")


class TestShowSymbolInfo:
    """Tests for _show_symbol_info helper function."""

    def test_displays_symbol_info(self, tmp_path: Path) -> None:
        """Test _show_symbol_info displays correct info."""
        from liq.data.cli.info import _show_symbol_info

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

        # Just verify it doesn't raise
        _show_symbol_info(store, "oanda/EUR_USD/bars/1m", "oanda", "EUR_USD", "1m")


class TestAppConfiguration:
    """Tests for app configuration."""

    def test_app_has_help(self) -> None:
        """Test app has help text."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "liq-data" in result.output or "Market data" in result.output

    def test_no_args_shows_help(self) -> None:
        """Test no arguments shows help (exit code 0 with no_args_is_help)."""
        result = runner.invoke(app, [])

        # no_args_is_help=True shows help but returns exit code 0
        assert "Usage" in result.output or "fetch" in result.output.lower()


class TestDeleteCommand:
    """Tests for the delete command."""

    def test_delete_removes_existing_data(self, tmp_path: Path) -> None:
        """Test delete command removes existing data via liq-store."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)
        assert store.exists("oanda/EUR_USD/bars/1m")

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["delete", "oanda", "EUR_USD", "--force"])

            assert result.exit_code == 0
            assert not store.exists("oanda/EUR_USD/bars/1m")
            assert "Deleted" in result.output

    def test_delete_nonexistent_data_shows_warning(self, tmp_path: Path) -> None:
        """Test delete command shows warning for non-existent data."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["delete", "oanda", "EUR_USD", "--force"])

            assert result.exit_code == 0
            assert "not found" in result.output.lower() or "No data" in result.output

    def test_delete_with_confirmation_prompt_yes(self, tmp_path: Path) -> None:
        """Test delete command with confirmation prompt (user confirms)."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            # Simulate user typing 'y' for confirmation
            result = runner.invoke(app, ["delete", "oanda", "EUR_USD"], input="y\n")

            assert result.exit_code == 0
            assert not store.exists("oanda/EUR_USD/bars/1m")

    def test_delete_with_confirmation_prompt_no(self, tmp_path: Path) -> None:
        """Test delete command with confirmation prompt (user declines)."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            # Simulate user typing 'n' for decline
            result = runner.invoke(app, ["delete", "oanda", "EUR_USD"], input="n\n")

            assert result.exit_code == 0
            assert store.exists("oanda/EUR_USD/bars/1m")  # Data should still exist
            assert "Cancelled" in result.output or "Aborted" in result.output

    def test_delete_force_skips_confirmation(self, tmp_path: Path) -> None:
        """Test delete command with --force skips confirmation."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            # No input needed with --force
            result = runner.invoke(app, ["delete", "oanda", "EUR_USD", "--force"])

            assert result.exit_code == 0
            assert not store.exists("oanda/EUR_USD/bars/1m")

    def test_delete_with_timeframe(self, tmp_path: Path) -> None:
        """Test delete command with specific timeframe."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)
        store.write("oanda/EUR_USD/bars/1h", df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["delete", "oanda", "EUR_USD", "--timeframe", "1h", "--force"]
            )

            assert result.exit_code == 0
            assert not store.exists("oanda/EUR_USD/bars/1h")
            assert store.exists("oanda/EUR_USD/bars/1m")  # 1m should still exist


class TestAuditCommand:
    """Tests for the audit command."""

    def test_audit_detects_gaps(self, tmp_path: Path) -> None:
        """Test audit command detects gaps in data."""
        store = ParquetStore(str(tmp_path))

        # Create data with a gap (missing 10:2)
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 1),
                # Gap here - missing 10:2
                datetime(2024, 1, 1, 10, 5),
            ],
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [100.0, 200.0, 150.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["audit", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            # Should report gaps or quality metrics
            assert "Audit" in result.output or "Gap" in result.output or "Quality" in result.output

    def test_audit_reports_quality_metrics(self, tmp_path: Path) -> None:
        """Test audit command reports quality metrics."""
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

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["audit", "oanda", "EUR_USD"])

            assert result.exit_code == 0
            # Should show some quality information
            assert "Audit" in result.output or "Total" in result.output

    def test_audit_data_not_found(self, tmp_path: Path) -> None:
        """Test audit command when data doesn't exist."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["audit", "oanda", "EUR_USD"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_audit_dry_run(self, tmp_path: Path) -> None:
        """Test audit command with --dry-run shows what would be done."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["audit", "oanda", "EUR_USD", "--dry-run"])

            assert result.exit_code == 0

    def test_audit_shows_gap_details_and_summary(self) -> None:
        df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 10),
            ],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [100.0, 200.0],
        })
        store = MagicMock()
        store.exists.return_value = True
        store.read.return_value = df

        with patch("liq.data.cli.validate.get_store", return_value=store):
            result = runner.invoke(app, ["audit", "oanda", "EUR_USD"])

        assert result.exit_code == 0
        assert "Gap Details" in result.output
        assert "Issues found" in result.output


class TestHealthReportCommand:
    """Tests for the health-report command."""

    def test_health_report_all_symbols(self, tmp_path: Path) -> None:
        """Test health-report command shows all available symbols."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)
        store.write("oanda/GBP_USD/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["health-report"])

            assert result.exit_code == 0
            assert "EUR_USD" in result.output or "Health" in result.output

    def test_health_report_single_provider(self, tmp_path: Path) -> None:
        """Test health-report command with --provider filter."""
        store = ParquetStore(str(tmp_path))

        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", df)
        store.write("binance/BTC_USDT/bars/1m", df)

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["health-report", "--provider", "oanda"])

            assert result.exit_code == 0
            # Should filter to only oanda
            assert "oanda" in result.output.lower() or "EUR_USD" in result.output

    def test_health_report_empty_data_directory(self, tmp_path: Path) -> None:
        """Test health-report command with no data."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["health-report"])

            assert result.exit_code == 0
            assert "No data" in result.output or "empty" in result.output.lower()

    def test_health_report_shows_summary(self, tmp_path: Path) -> None:
        """Test health-report command shows summary statistics."""
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

        with patch("liq.data.cli.validate.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(app, ["health-report"])

            assert result.exit_code == 0
            # Should have some summary or status info
            assert "Health" in result.output or "Report" in result.output or "oanda" in result.output

    def test_health_report_skips_bad_keys_and_errors(self) -> None:
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [100.0],
        })
        store = MagicMock()
        store.list_keys.return_value = ["badkey", "oanda/EUR_USD/bars/1m", "broken/EUR_USD/bars/1m"]

        def read_side_effect(key: str) -> pl.DataFrame:
            if key.startswith("broken/"):
                raise pl.exceptions.ComputeError("bad data")
            return df

        store.read.side_effect = read_side_effect
        store.get_date_range.return_value = (date(2024, 1, 1), date(2024, 1, 1))

        with patch("liq.data.cli.validate.get_store", return_value=store):
            result = runner.invoke(app, ["health-report"])

        assert result.exit_code == 0
        assert "Health Report" in result.output


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_same_symbol_different_providers(self, tmp_path: Path) -> None:
        """Test comparing same symbol across different providers."""
        store = ParquetStore(str(tmp_path))

        oanda_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1000, 1.1010],
            "high": [1.1050, 1.1060],
            "low": [1.0950, 1.0960],
            "close": [1.1020, 1.1030],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        # Similar data with slight differences
        polygon_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1001, 1.1011],
            "high": [1.1051, 1.1061],
            "low": [1.0951, 1.0961],
            "close": [1.1021, 1.1031],
            "volume": [110.0, 210.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD"]
            )

            assert result.exit_code == 0
            # Should show comparison statistics
            assert "Comparison" in result.output or "compare" in result.output.lower()

    def test_compare_different_symbols_same_provider(self, tmp_path: Path) -> None:
        """Test comparing different symbols from the same provider."""
        store = ParquetStore(str(tmp_path))

        eur_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1000, 1.1010],
            "high": [1.1050, 1.1060],
            "low": [1.0950, 1.0960],
            "close": [1.1020, 1.1030],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", eur_df)

        gbp_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.2700, 1.2710],
            "high": [1.2750, 1.2760],
            "low": [1.2650, 1.2660],
            "close": [1.2720, 1.2730],
            "volume": [150.0, 250.0],
        })
        store.write("oanda/GBP_USD/bars/1m", gbp_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "oanda:GBP_USD"]
            )

            assert result.exit_code == 0
            assert "Comparison" in result.output or "compare" in result.output.lower()

    def test_compare_handles_missing_first_data(self, tmp_path: Path) -> None:
        """Test compare command when first data source is missing."""
        store = ParquetStore(str(tmp_path))

        # Only create second data source
        polygon_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0)],
            "open": [1.1001],
            "high": [1.1051],
            "low": [1.0951],
            "close": [1.1021],
            "volume": [110.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD"]
            )

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_compare_handles_missing_second_data(self, tmp_path: Path) -> None:
        """Test compare command when second data source is missing."""
        store = ParquetStore(str(tmp_path))

        # Only create first data source
        oanda_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0)],
            "open": [1.1000],
            "high": [1.1050],
            "low": [1.0950],
            "close": [1.1020],
            "volume": [100.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD"]
            )

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_compare_aligns_timestamps(self, tmp_path: Path) -> None:
        """Test compare command aligns timestamps correctly (only common timestamps)."""
        store = ParquetStore(str(tmp_path))

        # Oanda has 3 timestamps
        oanda_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 1),
                datetime(2024, 1, 1, 10, 2),
            ],
            "open": [1.1000, 1.1010, 1.1020],
            "high": [1.1050, 1.1060, 1.1070],
            "low": [1.0950, 1.0960, 1.0970],
            "close": [1.1020, 1.1030, 1.1040],
            "volume": [100.0, 200.0, 300.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        # Polygon has only 2 overlapping timestamps
        polygon_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 10, 1),  # Only overlaps with middle
                datetime(2024, 1, 1, 10, 2),  # Only overlaps with last
            ],
            "open": [1.1011, 1.1021],
            "high": [1.1061, 1.1071],
            "low": [1.0961, 1.0971],
            "close": [1.1031, 1.1041],
            "volume": [210.0, 310.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD"]
            )

            assert result.exit_code == 0
            # Should mention aligned/matched bars count
            assert "2" in result.output or "aligned" in result.output.lower() or "matched" in result.output.lower()

    def test_compare_calculates_statistics(self, tmp_path: Path) -> None:
        """Test compare command calculates correlation and difference statistics."""
        store = ParquetStore(str(tmp_path))

        # Create data with known differences
        oanda_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 1),
                datetime(2024, 1, 1, 10, 2),
                datetime(2024, 1, 1, 10, 3),
            ],
            "open": [1.1000, 1.1010, 1.1020, 1.1030],
            "high": [1.1050, 1.1060, 1.1070, 1.1080],
            "low": [1.0950, 1.0960, 1.0970, 1.0980],
            "close": [1.1020, 1.1030, 1.1040, 1.1050],
            "volume": [100.0, 200.0, 300.0, 400.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        # Polygon with slightly different values
        polygon_df = pl.DataFrame({
            "timestamp": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 1),
                datetime(2024, 1, 1, 10, 2),
                datetime(2024, 1, 1, 10, 3),
            ],
            "open": [1.1001, 1.1011, 1.1021, 1.1031],
            "high": [1.1051, 1.1061, 1.1071, 1.1081],
            "low": [1.0951, 1.0961, 1.0971, 1.0981],
            "close": [1.1021, 1.1031, 1.1041, 1.1051],
            "volume": [105.0, 205.0, 305.0, 405.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD"]
            )

            assert result.exit_code == 0
            # Should show some statistics (correlation, mean diff, etc.)
            # At least one of these should appear in output
            has_stats = any(
                term in result.output.lower()
                for term in ["correlation", "diff", "mean", "max", "std", "statistics"]
            )
            assert has_stats, f"No statistics found in output: {result.output}"

    def test_compare_invalid_source_format(self, tmp_path: Path) -> None:
        """Test compare command rejects invalid source format."""
        store = ParquetStore(str(tmp_path))

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            # Missing colon separator
            result = runner.invoke(
                app, ["compare", "oanda_EUR_USD", "polygon:EUR_USD"]
            )

            assert result.exit_code == 1
            assert "format" in result.output.lower() or "invalid" in result.output.lower()

    def test_compare_output_csv(self, tmp_path: Path) -> None:
        """Test compare command with --output option for CSV export."""
        store = ParquetStore(str(tmp_path))

        oanda_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1000, 1.1010],
            "high": [1.1050, 1.1060],
            "low": [1.0950, 1.0960],
            "close": [1.1020, 1.1030],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        polygon_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1001, 1.1011],
            "high": [1.1051, 1.1061],
            "low": [1.0951, 1.0961],
            "close": [1.1021, 1.1031],
            "volume": [110.0, 210.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        output_file = tmp_path / "comparison.csv"

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app,
                ["compare", "oanda:EUR_USD", "polygon:EUR_USD", "--output", str(output_file)],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            # CSV should have content
            content = output_file.read_text()
            assert "timestamp" in content.lower() or "," in content

    def test_compare_output_json(self, tmp_path: Path) -> None:
        """Test compare command with --output option for JSON export."""
        store = ParquetStore(str(tmp_path))

        oanda_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1000, 1.1010],
            "high": [1.1050, 1.1060],
            "low": [1.0950, 1.0960],
            "close": [1.1020, 1.1030],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        polygon_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1001, 1.1011],
            "high": [1.1051, 1.1061],
            "low": [1.0951, 1.0961],
            "close": [1.1021, 1.1031],
            "volume": [110.0, 210.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        output_file = tmp_path / "comparison.json"

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app,
                ["compare", "oanda:EUR_USD", "polygon:EUR_USD", "--output", str(output_file)],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            # JSON should be valid
            import json
            content = json.loads(output_file.read_text())
            assert isinstance(content, (dict, list))

    def test_compare_no_overlapping_timestamps(self, tmp_path: Path) -> None:
        """Test compare command when there are no overlapping timestamps."""
        store = ParquetStore(str(tmp_path))

        # Different time ranges with no overlap
        oanda_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1000, 1.1010],
            "high": [1.1050, 1.1060],
            "low": [1.0950, 1.0960],
            "close": [1.1020, 1.1030],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        polygon_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 2, 10, 1)],  # Next day
            "open": [1.1001, 1.1011],
            "high": [1.1051, 1.1061],
            "low": [1.0951, 1.0961],
            "close": [1.1021, 1.1031],
            "volume": [110.0, 210.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD"]
            )

            # Should fail or show warning about no overlap
            assert "no" in result.output.lower() or "overlap" in result.output.lower() or result.exit_code == 1

    def test_compare_with_timeframe_option(self, tmp_path: Path) -> None:
        """Test compare command with --timeframe option."""
        store = ParquetStore(str(tmp_path))

        # Create 1h data
        oanda_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0)],
            "open": [1.1000, 1.1010],
            "high": [1.1050, 1.1060],
            "low": [1.0950, 1.0960],
            "close": [1.1020, 1.1030],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1h", oanda_df)

        polygon_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 11, 0)],
            "open": [1.1001, 1.1011],
            "high": [1.1051, 1.1061],
            "low": [1.0951, 1.0961],
            "close": [1.1021, 1.1031],
            "volume": [110.0, 210.0],
        })
        store.write("polygon/EUR_USD/bars/1h", polygon_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD", "--timeframe", "1h"]
            )

            assert result.exit_code == 0
            assert "Comparison" in result.output or "compare" in result.output.lower()

    def test_compare_shows_price_column_differences(self, tmp_path: Path) -> None:
        """Test compare command shows differences for each price column."""
        store = ParquetStore(str(tmp_path))

        oanda_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1000, 1.1010],
            "high": [1.1050, 1.1060],
            "low": [1.0950, 1.0960],
            "close": [1.1020, 1.1030],
            "volume": [100.0, 200.0],
        })
        store.write("oanda/EUR_USD/bars/1m", oanda_df)

        polygon_df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)],
            "open": [1.1001, 1.1011],
            "high": [1.1051, 1.1061],
            "low": [1.0951, 1.0961],
            "close": [1.1021, 1.1031],
            "volume": [110.0, 210.0],
        })
        store.write("polygon/EUR_USD/bars/1m", polygon_df)

        with patch("liq.data.cli.manage.get_store") as mock_store:
            mock_store.return_value = store

            result = runner.invoke(
                app, ["compare", "oanda:EUR_USD", "polygon:EUR_USD"]
            )

            assert result.exit_code == 0
            # Should show differences for price columns (at least one)
            has_column_info = any(
                col in result.output.lower()
                for col in ["open", "high", "low", "close"]
            )
            assert has_column_info, f"No price column info in output: {result.output}"

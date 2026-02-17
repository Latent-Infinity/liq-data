from datetime import UTC
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from liq.data.cli_qa import app
from liq.data.settings import get_settings
from liq.store.parquet import ParquetStore

runner = CliRunner()


def test_cli_qa_storage(tmp_path: Path) -> None:
    """QA should read data from liq-store-managed storage."""
    from datetime import datetime

    store = ParquetStore(str(tmp_path))
    storage_key = "oanda/EUR_USD/bars/1m"

    df = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
        ],
        "open": [1.0, 1.5],
        "high": [2.0, 2.5],
        "low": [0.5, 1.0],
        "close": [1.5, 2.0],
        "volume": [10.0, 20.0],
    })
    store.write(storage_key, df, mode="overwrite")

    get_settings.cache_clear()
    result = runner.invoke(app, [storage_key], env={"DATA_ROOT": str(tmp_path)})
    assert result.exit_code == 0


def test_cli_qa_invalid_key() -> None:
    """Invalid storage key should error."""
    result = runner.invoke(app, ["invalid/key"])
    assert result.exit_code != 0

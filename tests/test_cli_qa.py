import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from liq.data.cli_qa import app

runner = CliRunner()


def test_cli_qa_json(tmp_path: Path) -> None:
    bars = [
        {"timestamp": "2024-01-01T00:00:00Z", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
        {"timestamp": "2024-01-01T00:01:00Z", "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 20},
    ]
    path = tmp_path / "bars.json"
    path.write_text(json.dumps(bars))
    result = runner.invoke(app, [str(path)])
    assert result.exit_code == 0
    assert "Missing ratio" in result.stdout or "Missing ratio" in result.output


def test_cli_qa_parquet(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    df = pl.DataFrame({
        "timestamp": [
            datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        ],
        "open": [1.0, 1.5],
        "high": [2.0, 2.5],
        "low": [0.5, 1.0],
        "close": [1.5, 2.0],
        "volume": [10.0, 20.0],
    })
    path = tmp_path / "bars.parquet"
    df.write_parquet(path)
    result = runner.invoke(app, [str(path)])
    assert result.exit_code == 0

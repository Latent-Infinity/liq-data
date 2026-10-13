from datetime import date
from pathlib import Path

import polars as pl
import pytest

from liq.data.exceptions import SchemaValidationError, UndeclaredReadError
from liq.data.service import BarCoverageObservation, DataService, StoredTimestampAudit


def _audit(
    service: DataService,
    *,
    batch_size: int = 10,
    max_rows: int = 100,
) -> StoredTimestampAudit:
    return service.audit_stored_timestamps(
        "tradestation",
        "metadata-only",
        "1d",
        date(2019, 1, 1),
        date(2019, 1, 2),
        purpose="dev_smoke",
        arm_id="timestamp_schema_test",
        batch_size=batch_size,
        max_rows=max_rows,
    )


@pytest.mark.parametrize("batch_size,max_rows", [(0, 1), (1, 0), (-1, 1), (1, -1), (True, 1)])
def test_invalid_budget_refuses_before_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    max_rows: int,
) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(service.store, "exists", lambda *_: pytest.fail("store reached"))
    with pytest.raises(ValueError, match="positive integers"):
        _audit(service, batch_size=batch_size, max_rows=max_rows)


def test_reversed_dates_refuse_before_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(service.store, "exists", lambda *_: pytest.fail("store reached"))
    with pytest.raises(ValueError, match="start"):
        service.audit_stored_timestamps(
            "tradestation",
            "metadata-only",
            "1d",
            date(2019, 1, 2),
            date(2019, 1, 1),
            purpose="dev_smoke",
            arm_id="timestamp_schema_test",
            batch_size=1,
            max_rows=1,
        )


@pytest.mark.parametrize("purpose,error", [(None, UndeclaredReadError), ("", ValueError)])
def test_absent_and_unknown_purpose_refuse_before_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    purpose: str | None,
    error: type[Exception],
) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(service.store, "exists", lambda *_: pytest.fail("store reached"))
    with pytest.raises(error):
        service.audit_stored_timestamps(
            "tradestation",
            "metadata-only",
            "1d",
            date(2019, 1, 1),
            date(2019, 1, 2),
            purpose=purpose,
            arm_id="timestamp_schema_test",
            batch_size=1,
            max_rows=1,
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"timestamp": pl.Datetime("us", "UTC"), "unexpected": pl.String},
        {"timestamp": pl.String},
        {"timestamp": pl.Datetime("us")},
        {"timestamp": pl.Datetime("us", "America/New_York")},
    ],
)
def test_invalid_empty_schema_refuses_without_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema,
) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(
        service, "iter_batches", lambda *_args, **_kwargs: iter([pl.DataFrame(schema=schema)])
    )
    with pytest.raises(SchemaValidationError):
        _audit(service)


def test_null_timestamp_is_invalid_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DataService(data_root=tmp_path)
    invalid_metadata = pl.DataFrame(
        {"timestamp": [None]}, schema={"timestamp": pl.Datetime("us", "UTC")}
    )
    monkeypatch.setattr(service, "iter_batches", lambda *_args, **_kwargs: iter([invalid_metadata]))
    with pytest.raises(SchemaValidationError, match="null timestamps"):
        _audit(service)


def test_no_batches_cannot_certify_an_empty_timestamp_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(service, "iter_batches", lambda *_args, **_kwargs: iter(()))
    with pytest.raises(SchemaValidationError):
        _audit(service)


@pytest.mark.parametrize("timezone", ["UTC", "America/New_York"])
def test_real_empty_parquet_refuses_unobserved_schema(tmp_path: Path, timezone: str) -> None:
    path = tmp_path / "tradestation/metadata-only/bars/1d/year=2019/schema-only.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(schema={"timestamp": pl.Datetime("us", timezone)}).write_parquet(path)
    with pytest.raises(SchemaValidationError):
        _audit(DataService(data_root=tmp_path))


def test_source_anomaly_metadata_keeps_distinct_reasons() -> None:
    coverage = BarCoverageObservation(
        provider="metadata",
        symbol="metadata",
        timeframe="1d",
        requested_start="2019-01-01",
        requested_end="2019-01-02",
        row_count=3,
        unique_timestamp_count=2,
        duplicate_timestamp_count=1,
        outside_requested_window_count=1,
        first_timestamp=None,
        last_timestamp=None,
        timestamp_sha256="0" * 64,
    )
    audit = StoredTimestampAudit(coverage=coverage, utc_date_counts=(("2019-01-01", 3),))
    assert audit.anomalies == (
        "duplicate_timestamps",
        "out_of_scope_timestamps",
        "multiple_daily_rows_per_utc_date",
    )

from datetime import date
from pathlib import Path

import pytest

from liq.data.exceptions import LockboxViolationError, SchemaValidationError, UndeclaredReadError
from liq.data.service import DataService
from liq.store.timestamp_metadata import PinnedParquetFile, TimestampMetadataRefused


def test_invalid_budget_refuses_before_tree_enumeration(tmp_path: Path, monkeypatch) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(Path, "rglob", lambda *_: pytest.fail("invalid budget reached storage"))
    with pytest.raises(TimestampMetadataRefused, match="budget"):
        service.preflight_stored_timestamps(
            "tradestation",
            "metadata",
            "1d",
            date(2019, 1, 1),
            date(2019, 1, 2),
            files=(),
            purpose="dev_smoke",
            arm_id="metadata-schema-test",
            max_files=0,
            max_file_bytes=100000,
            max_row_groups=10,
        )


def test_unlisted_directory_refuses_before_footer_inspection(tmp_path: Path, monkeypatch) -> None:
    service = DataService(data_root=tmp_path)
    base = tmp_path / "tradestation/metadata/bars/1d"
    (base / "unexpected").mkdir(parents=True)
    monkeypatch.setattr(
        "liq.data.service.inspect_timestamp_metadata",
        lambda *_args, **_kwargs: pytest.fail("unexpected tree reached footer parser"),
    )
    with pytest.raises(SchemaValidationError, match="inventory"):
        service.preflight_stored_timestamps(
            "tradestation",
            "metadata",
            "1d",
            date(2019, 1, 1),
            date(2019, 1, 2),
            files=(),
            purpose="dev_smoke",
            arm_id="metadata-schema-test",
            max_files=1,
            max_file_bytes=100000,
            max_row_groups=10,
        )


def test_undeclared_footer_inspection_refuses_before_storage(tmp_path: Path, monkeypatch) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(
        service.store, "exists", lambda *_: pytest.fail("undeclared storage access")
    )
    with pytest.raises(UndeclaredReadError):
        service.preflight_stored_timestamps(
            "tradestation",
            "metadata",
            "1d",
            date(2019, 1, 1),
            date(2019, 1, 2),
            files=(),
            purpose=None,
            arm_id="metadata-schema-test",
            max_files=1,
            max_file_bytes=100000,
            max_row_groups=10,
        )
    assert list(tmp_path.iterdir()) == []


def test_protected_footer_scope_refuses_before_storage(tmp_path: Path, monkeypatch) -> None:
    service = DataService(data_root=tmp_path)
    monkeypatch.setattr(service.store, "exists", lambda *_: pytest.fail("protected storage access"))
    with pytest.raises(LockboxViolationError):
        service.preflight_stored_timestamps(
            "tradestation",
            "metadata",
            "1d",
            date(2026, 1, 1),
            date(2026, 1, 2),
            files=(),
            purpose="discovery",
            arm_id="metadata-schema-test",
            max_files=1,
            max_file_bytes=100000,
            max_row_groups=10,
        )


def test_file_identity_cannot_expand_declared_key(tmp_path: Path) -> None:
    service = DataService(data_root=tmp_path)
    with pytest.raises(ValueError, match="key"):
        service.preflight_stored_timestamps(
            "tradestation",
            "metadata",
            "1d",
            date(2019, 1, 1),
            date(2019, 1, 2),
            files=(PinnedParquetFile("tradestation/other/bars/1d/a.parquet", "0" * 64),),
            purpose="dev_smoke",
            arm_id="metadata-schema-test",
            max_files=1,
            max_file_bytes=100000,
            max_row_groups=10,
        )

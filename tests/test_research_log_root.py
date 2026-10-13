from datetime import date
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from liq.data.service import DataService
from liq.store.timestamp_metadata import PinnedParquetFile, TimestampMetadataRefused


def test_default_research_log_location_is_unchanged(tmp_path: Path) -> None:
    service = DataService(data_root=tmp_path)
    assert service.lockbox_guard.usage_log_path == tmp_path / "lockbox_usage_log.jsonl"


def test_readonly_source_logs_to_separate_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = source / "tradestation/metadata/bars/1d/empty.parquet"
    target.parent.mkdir(parents=True)
    pq.write_table(pa.table({"timestamp": pa.array([], type=pa.timestamp("us", tz="UTC"))}), target)
    identity = PinnedParquetFile(
        target.relative_to(source).as_posix(), sha256(target.read_bytes()).hexdigest()
    )
    logs = tmp_path / "audit"
    source.chmod(0o555)
    try:
        service = DataService(data_root=source, research_log_root=logs)
        with pytest.raises(TimestampMetadataRefused, match="empty"):
            service.preflight_stored_timestamps(
                "tradestation",
                "metadata",
                "1d",
                date(2019, 1, 1),
                date(2019, 1, 2),
                files=(identity,),
                purpose="dev_smoke",
                arm_id="metadata-schema-test",
                max_files=1,
                max_file_bytes=100000,
                max_row_groups=10,
            )
        assert service.lockbox_guard.usage_log_path == logs / "lockbox_usage_log.jsonl"
        assert (logs / "lockbox_usage_log.jsonl").is_file()
        assert (logs / "timestamp_metadata_usage.jsonl").is_file()
        assert not list(source.glob("*.jsonl"))
        assert sha256(target.read_bytes()).hexdigest() == identity.sha256
    finally:
        source.chmod(0o755)

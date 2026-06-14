"""Staging-dir contract for Databento batch downloads.

Batch artifacts (`.dbn.zst` files + the SDK's sibling
`manifest.json` / `condition.json`) are transient: ingested into
parquet bars + the manifest, then thrown away. The provider stages
each download into a per-call ``tempfile.TemporaryDirectory()`` so
the cleanup is guaranteed on success **and** on failure, no global
``_cache/`` folder needs to be reasoned about, and the raw artifacts
never leak into the repo root or persist past the parquet write.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    DatabentoBarRecord,
    DatabentoProvider,
)


def _record(symbol: str = "AAPL") -> DatabentoBarRecord:
    ts = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    return DatabentoBarRecord(
        ts_event_ns=int(ts.timestamp() * 1e9),
        instrument_id=12345,
        open_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        high_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        low_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        close_q9=int(Decimal("100") * DATABENTO_PRICE_SCALE),
        volume=1_000,
        symbol=symbol,
    )


class _FakeDBNStore:
    def __init__(self, records: list[DatabentoBarRecord]) -> None:
        self._records = records
        self.symbology = {
            "mappings": {
                r.symbol: [
                    {
                        "start_date": "2025-01-01",
                        "end_date": "2099-12-31",
                        "symbol": str(r.instrument_id),
                    }
                ]
                for r in records
            }
        }

        class _Meta:
            schema = "ohlcv-1m"

        self.metadata = _Meta()

    def __iter__(self):
        return iter(self._records)


class _BatchDropsRealFiles:
    """Pretend to be the real SDK: write actual ``.dbn.zst`` content
    into ``output_dir`` and return the resulting ``list[Path]``."""

    def __init__(self, payload_records: list[DatabentoBarRecord]) -> None:
        self._payload_records = payload_records
        self.output_dirs_seen: list[str] = []
        self.staged_paths_seen: list[Path] = []
        self.download_calls = 0

    def submit_job(self, **kwargs: Any) -> dict[str, Any]:
        return {"id": "job-stage", "state": "received"}

    def get_job_details(self, job_id: str) -> dict[str, Any]:
        return {"id": job_id, "state": "done"}

    def download(self, *, job_id: str, output_dir: str, **_kwargs: Any) -> list[Path]:
        self.download_calls += 1
        self.output_dirs_seen.append(output_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Drop a real fake file path; the from_file monkeypatch maps
        # it back to an in-memory store.
        path = out / "equs-mini-20250102.ohlcv-1m.dbn.zst"
        path.write_bytes(b"fake-dbn-payload")
        # Sibling JSON files the real SDK includes — should also be
        # cleaned up via the temp-dir.
        (out / "condition.json").write_text("{}", encoding="utf-8")
        (out / "manifest.json").write_text("{}", encoding="utf-8")
        self.staged_paths_seen.append(path)
        return [path, out / "condition.json", out / "manifest.json"]


class _BatchFailsDownload(_BatchDropsRealFiles):
    """Same shape, but ``download`` writes the file then raises so the
    cleanup path runs through the exception branch."""

    def download(self, *, job_id: str, output_dir: str, **_kwargs: Any) -> list[Path]:
        # Stage some partial files first so we have something to verify
        # gets cleaned up on the failure path.
        super().download(job_id=job_id, output_dir=output_dir)
        raise RuntimeError("simulated download failure")


class _Timeseries:
    def get_range(self, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("get_range must not run for batch tests")


class _Client:
    def __init__(self, batch: Any) -> None:
        self.timeseries = _Timeseries()
        self.batch = batch


@pytest.fixture
def patched_from_file(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[Path, _FakeDBNStore]:
    """Monkey-patch ``databento.DBNStore.from_file`` to look up an
    in-memory store keyed by the path the SDK stages."""
    import databento as db  # noqa: PLC0415 — real import target

    table: dict[Path, _FakeDBNStore] = {}

    def _from_file(path: Any, *_a: Any, **_k: Any) -> _FakeDBNStore:
        return table[Path(str(path))]

    monkeypatch.setattr(db.DBNStore, "from_file", staticmethod(_from_file))
    return table


class TestStagingDirCleanup:
    def test_success_removes_staging_dir(
        self,
        tmp_path: Path,
        patched_from_file: dict[Path, _FakeDBNStore],
    ) -> None:
        batch = _BatchDropsRealFiles([_record()])
        provider = DatabentoProvider(
            api_key="x",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )

        # When the SDK calls from_file, return a store with the records.
        # We seed the table only after we know the path — but the path
        # is generated lazily, so register a side-effect on the batch.
        def _register_path(path: Path) -> None:
            patched_from_file[path] = _FakeDBNStore([_record()])

        original_download = batch.download

        def _download_and_register(**kwargs: Any) -> list[Path]:
            paths = original_download(**kwargs)
            for p in paths:
                if str(p).endswith(".dbn.zst"):
                    _register_path(p)
            return paths

        batch.download = _download_and_register  # type: ignore[method-assign]

        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        # The SDK was handed a path. After fetch_bars returns, that path
        # must no longer exist (the TemporaryDirectory cleaned up).
        assert batch.output_dirs_seen, "download was never called"
        for staged in batch.output_dirs_seen:
            assert not os.path.exists(staged), (
                f"staging dir leaked after successful fetch: {staged}"
            )

    def test_download_failure_still_removes_staging_dir(self, tmp_path: Path) -> None:
        batch = _BatchFailsDownload([_record()])
        provider = DatabentoProvider(
            api_key="x",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
            max_retry_attempts=1,
        )

        with pytest.raises(RuntimeError, match="simulated download failure"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        # Even though the download raised mid-flight, the per-call
        # TemporaryDirectory must have been removed.
        assert batch.output_dirs_seen, "download was never called"
        for staged in batch.output_dirs_seen:
            assert not os.path.exists(staged), f"staging dir leaked after failed fetch: {staged}"

    def test_marker_survives_download_failure_for_resume(self, tmp_path: Path) -> None:
        """The download files get cleaned up, but the marker stays so
        a retry can pick up where we left off without re-billing."""
        batch = _BatchFailsDownload([_record()])
        jobs_dir = tmp_path / "jobs"
        provider = DatabentoProvider(
            api_key="x",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(jobs_dir),
            sleep_fn=lambda _s: None,
            max_retry_attempts=1,
        )
        with pytest.raises(RuntimeError):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        markers = list(jobs_dir.glob("*.json"))
        assert markers, "resume marker was deleted along with staging dir"


class TestNoCwdLeakage:
    def test_batch_download_does_not_touch_cwd(
        self, tmp_path: Path, patched_from_file: dict[Path, _FakeDBNStore]
    ) -> None:
        batch = _BatchDropsRealFiles([_record()])
        provider = DatabentoProvider(
            api_key="x",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )

        original_download = batch.download

        def _download_and_register(**kwargs: Any) -> list[Path]:
            paths = original_download(**kwargs)
            for p in paths:
                if str(p).endswith(".dbn.zst"):
                    patched_from_file[p] = _FakeDBNStore([_record()])
            return paths

        batch.download = _download_and_register  # type: ignore[method-assign]

        cwd_before = {p.name for p in Path.cwd().iterdir() if p.name.startswith("EQUS-")}
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        cwd_after = {p.name for p in Path.cwd().iterdir() if p.name.startswith("EQUS-")}
        assert cwd_after == cwd_before, "EQUS-* artifacts leaked into CWD"

"""Regression tests for the real ``batch.download`` return shape.

The real Databento SDK's ``HistoricalBatch.download(...)`` returns
``list[Path]`` of extracted ``.dbn`` / ``.dbn.zst`` files — *not* a
single DBN store. Earlier in-tests fakes returned a store directly,
so the provider iterated the path list as if it were a store and
hit ``AttributeError`` on the first ``Path``.

These tests pin the fixed shape: when ``batch.download`` returns a
list of paths, the provider opens each via
``databento.DBNStore.from_file(...)`` and unifies iteration +
symbology across the resulting stores.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from liq.data.providers.databento import (
    DATABENTO_PRICE_SCALE,
    DatabentoBarRecord,
    DatabentoProvider,
)

# A sentinel path the fake batch returns; never opened for real because
# the test monkeypatches ``DBNStore.from_file``.
_SENTINEL_A = Path("/tmp/fake-batch/job-1/output_a.dbn.zst")
_SENTINEL_B = Path("/tmp/fake-batch/job-1/output_b.dbn.zst")


def _record(symbol: str, day: int, *, minute: int = 0) -> DatabentoBarRecord:
    ts = datetime(2025, 1, day, 14, 30 + minute, tzinfo=UTC)
    return DatabentoBarRecord(
        ts_event_ns=int(ts.timestamp() * 1e9),
        instrument_id=12345 if symbol == "AAPL" else 15144,
        open_q9=int(100 * DATABENTO_PRICE_SCALE),
        high_q9=int(100 * DATABENTO_PRICE_SCALE),
        low_q9=int(100 * DATABENTO_PRICE_SCALE),
        close_q9=int(100 * DATABENTO_PRICE_SCALE),
        volume=1_000,
        symbol=symbol,
    )


class _FakeDBNStore:
    """Per-file fake — represents one ``.dbn.zst`` extracted from the
    batch zip. Holds its own symbology slice and record stream."""

    def __init__(
        self,
        records: list[DatabentoBarRecord],
        symbology: dict[str, Any],
        *,
        schema: str = "ohlcv-1m",
    ) -> None:
        self._records = records
        self.symbology = symbology

        class _Meta:
            pass

        self.metadata = _Meta()
        self.metadata.schema = schema

    def __iter__(self) -> Iterable[DatabentoBarRecord]:
        return iter(self._records)

    def close(self) -> None:
        return None


class _BatchReturningPaths:
    """Mimic the real Databento batch API: ``download`` returns
    ``list[Path]`` rather than a single store."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self.submit_calls: list[dict] = []
        self.download_calls: list[dict] = []

    def submit_job(self, **kwargs: Any) -> dict[str, Any]:
        self.submit_calls.append(kwargs)
        return {"id": "job-1", "state": "received"}

    def get_job_details(self, job_id: str) -> dict[str, Any]:
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs: Any) -> list[Path]:
        self.download_calls.append(kwargs)
        return list(self._paths)


class _BatchReturningSinglePath(_BatchReturningPaths):
    def __init__(self, path: Path) -> None:
        super().__init__([path])
        self._path = path

    def download(self, **kwargs: Any) -> Path:
        self.download_calls.append(kwargs)
        return self._path


class _BatchReturningDirectory(_BatchReturningPaths):
    def __init__(self, directory: Path) -> None:
        super().__init__([directory])
        self._directory = directory

    def download(self, **kwargs: Any) -> Path:
        self.download_calls.append(kwargs)
        return self._directory


class _Timeseries:
    def get_range(self, **kwargs: Any) -> Any:  # pragma: no cover — batch path only
        raise AssertionError("get_range must not be called when the span routes to batch")


class _Client:
    def __init__(self, batch: Any) -> None:
        self.timeseries = _Timeseries()
        self.batch = batch


@pytest.fixture
def file_store_table(monkeypatch: pytest.MonkeyPatch) -> dict[Path, _FakeDBNStore]:
    """Build a path → DBNStore lookup that ``DBNStore.from_file`` will
    consult; lets the test author per-file slices of records + symbology."""
    table: dict[Path, _FakeDBNStore] = {}

    def _from_file(path: Any, *_a: Any, **_k: Any) -> _FakeDBNStore:
        p = Path(str(path))
        if p not in table:
            raise AssertionError(f"unexpected DBNStore.from_file path: {p}")
        return table[p]

    import databento as db  # noqa: PLC0415 — real-shape patch target

    monkeypatch.setattr(db.DBNStore, "from_file", staticmethod(_from_file))
    return table


@pytest.fixture
def provider_for_batch(tmp_path: Path) -> tuple[DatabentoProvider, _BatchReturningPaths]:
    batch = _BatchReturningPaths([_SENTINEL_A, _SENTINEL_B])
    provider = DatabentoProvider(
        api_key="cassette",
        client=_Client(batch),
        batch_threshold_days=1,
        batch_jobs_dir=str(tmp_path / "jobs"),
        sleep_fn=lambda _s: None,
    )
    return provider, batch


# ----- regression tests -----------------------------------------------------


class TestBatchPathListShape:
    def test_provider_opens_each_returned_dbn_file(
        self,
        provider_for_batch: tuple[DatabentoProvider, _BatchReturningPaths],
        file_store_table: dict[Path, _FakeDBNStore],
    ) -> None:
        provider, _batch = provider_for_batch
        # Split the records across two files (the real SDK chunks by
        # session count or job partition).
        file_store_table[_SENTINEL_A] = _FakeDBNStore(
            records=[_record("AAPL", 2), _record("AAPL", 2, minute=1)],
            symbology={
                "mappings": {
                    "AAPL": [
                        {
                            "start_date": "2025-01-02",
                            "end_date": "2025-01-03",
                            "symbol": "12345",
                        }
                    ]
                }
            },
        )
        file_store_table[_SENTINEL_B] = _FakeDBNStore(
            records=[_record("AAPL", 3)],
            symbology={
                "mappings": {
                    "AAPL": [
                        {
                            "start_date": "2025-01-03",
                            "end_date": "2025-01-04",
                            "symbol": "12345",
                        }
                    ]
                }
            },
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        # 2 records from file A + 1 from file B = 3.
        assert df.height == 3

    def test_provider_merges_symbology_across_files(
        self,
        provider_for_batch: tuple[DatabentoProvider, _BatchReturningPaths],
        file_store_table: dict[Path, _FakeDBNStore],
    ) -> None:
        provider, _batch = provider_for_batch
        file_store_table[_SENTINEL_A] = _FakeDBNStore(
            records=[_record("AAPL", 2)],
            symbology={
                "mappings": {
                    "AAPL": [
                        {
                            "start_date": "2025-01-02",
                            "end_date": "2025-01-03",
                            "symbol": "12345",
                        }
                    ]
                }
            },
        )
        file_store_table[_SENTINEL_B] = _FakeDBNStore(
            records=[_record("MSFT", 2)],
            symbology={
                "mappings": {
                    "MSFT": [
                        {
                            "start_date": "2025-01-02",
                            "end_date": "2025-01-03",
                            "symbol": "67890",
                        }
                    ]
                }
            },
        )

        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        # Both file symbologies must have been merged and (when a store
        # is attached) persisted under SYMBOLOGY_KEY. Here we just
        # confirm fetch did not crash on the merge.

    def test_non_dbn_files_in_path_list_are_ignored(
        self,
        provider_for_batch: tuple[DatabentoProvider, _BatchReturningPaths],
        file_store_table: dict[Path, _FakeDBNStore],
    ) -> None:
        """The real SDK includes a ``manifest.json`` and ``symbology.json``
        alongside the data files. Only ``.dbn`` / ``.dbn.zst`` /
        ``.dbn.gz`` should be opened as record streams."""
        provider, batch = provider_for_batch
        # Replace the sentinel list with a mix.
        manifest = Path("/tmp/fake-batch/job-1/manifest.json")
        batch._paths = [manifest, _SENTINEL_A]
        file_store_table[_SENTINEL_A] = _FakeDBNStore(
            records=[_record("AAPL", 2)],
            symbology={
                "mappings": {
                    "AAPL": [
                        {
                            "start_date": "2025-01-02",
                            "end_date": "2025-01-03",
                            "symbol": "12345",
                        }
                    ]
                }
            },
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert df.height == 1

    def test_single_path_download_result_is_opened(
        self,
        tmp_path: Path,
        file_store_table: dict[Path, _FakeDBNStore],
    ) -> None:
        batch = _BatchReturningSinglePath(_SENTINEL_A)
        provider = DatabentoProvider(
            api_key="cassette",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )
        file_store_table[_SENTINEL_A] = _FakeDBNStore(
            records=[_record("AAPL", 2)],
            symbology={
                "mappings": {
                    "AAPL": [
                        {
                            "start_date": "2025-01-02",
                            "end_date": "2025-01-03",
                            "symbol": "12345",
                        }
                    ]
                }
            },
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert df.height == 1

    def test_directory_download_result_discovers_dbn_files(
        self,
        tmp_path: Path,
        file_store_table: dict[Path, _FakeDBNStore],
    ) -> None:
        out = tmp_path / "sdk-output"
        dbn_path = out / "nested" / "output.dbn.zst"
        dbn_path.parent.mkdir(parents=True)
        dbn_path.write_bytes(b"placeholder")
        (out / "manifest.json").write_text("{}", encoding="utf-8")
        batch = _BatchReturningDirectory(out)
        provider = DatabentoProvider(
            api_key="cassette",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )
        file_store_table[dbn_path] = _FakeDBNStore(
            records=[_record("AAPL", 2)],
            symbology={
                "mappings": {
                    "AAPL": [
                        {
                            "start_date": "2025-01-02",
                            "end_date": "2025-01-03",
                            "symbol": "12345",
                        }
                    ]
                }
            },
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert df.height == 1

    def test_batch_chunks_are_materialized_one_file_at_a_time(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = [Path(f"/tmp/fake-batch/job-1/chunk_{i}.dbn.zst") for i in range(64)]
        state = {"open": 0, "max_open": 0}

        class _CountingStore(_FakeDBNStore):
            def __init__(self, path: Path) -> None:
                state["open"] += 1
                state["max_open"] = max(state["max_open"], state["open"])
                if state["open"] > 1:
                    raise OSError("too many open files")
                super().__init__(
                    records=[_record("AAPL", 2)],
                    symbology={
                        "mappings": {
                            "AAPL": [
                                {
                                    "start_date": "2025-01-02",
                                    "end_date": "2025-01-03",
                                    "symbol": str(path),
                                }
                            ]
                        }
                    },
                )

            def close(self) -> None:
                state["open"] -= 1

        def _from_file(path: Any, *_a: Any, **_k: Any) -> _CountingStore:
            return _CountingStore(Path(str(path)))

        import databento as db  # noqa: PLC0415 — real-shape patch target

        monkeypatch.setattr(db.DBNStore, "from_file", staticmethod(_from_file))
        batch = _BatchReturningPaths(paths)
        provider = DatabentoProvider(
            api_key="cassette",
            client=_Client(batch),
            batch_threshold_days=1,
            batch_jobs_dir=str(tmp_path / "jobs"),
            sleep_fn=lambda _s: None,
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        assert df.height == len(paths)
        assert state == {"open": 0, "max_open": 1}

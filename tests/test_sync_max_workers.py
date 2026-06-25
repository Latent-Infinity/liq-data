"""Tests for ``DataService.sync(..., max_workers=...)``.

``max_workers`` is a keyword-only parameter on :meth:`DataService.sync`.
The default is ``1`` (sequential); larger values run the per-symbol loop
in a bounded ``ThreadPoolExecutor``.

Cross-cohort acquisition can opt in to per-symbol parallelism within the
provider's concurrent-request envelope.

Tests pin:

* default sequential behaviour
* invalid ``max_workers`` rejection
* bounded concurrency at ``max_workers > 1`` (peak in-flight <= N)
* per-symbol transactional safety preserved
* per-symbol failure aborts the whole sync (existing contract)
* file lock semantics unchanged
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
from filelock import FileLock

from liq.data.manifest import CoverageManifest
from liq.data.service import DataService
from liq.data.sync_events import SyncLockedError
from liq.data.universes import UniverseDefinition, UniverseKind

# ----- helpers --------------------------------------------------------------


class _RecordingProvider:
    """Tracks per-symbol fetch_bars calls + concurrent in-flight count.

    Each call sleeps ``hold_seconds`` so concurrent execution is
    observable; the peak ``concurrent_calls`` counter is updated under a
    lock and surfaced for the test to assert ``max_workers`` is honoured.
    """

    def __init__(
        self,
        *,
        hold_seconds: float = 0.05,
        failing_symbols: Sequence[str] = (),
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.calls_lock = threading.Lock()
        self.in_flight = 0
        self.peak_concurrent = 0
        self.concurrent_lock = threading.Lock()
        self.hold_seconds = hold_seconds
        self.failing_symbols = set(failing_symbols)

    @property
    def name(self) -> str:
        return "databento"

    @property
    def supported_asset_classes(self) -> list[str]:
        return ["equity"]

    @property
    def supported_timeframes(self) -> list[str]:
        return ["1m"]

    def fetch_bars(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        timeframe: str,
    ) -> pl.DataFrame:
        with self.concurrent_lock:
            self.in_flight += 1
            self.peak_concurrent = max(self.peak_concurrent, self.in_flight)
        try:
            with self.calls_lock:
                self.calls.append(
                    {"symbol": symbol, "start": start, "end": end, "timeframe": timeframe}
                )
            time.sleep(self.hold_seconds)
            if symbol in self.failing_symbols:
                raise RuntimeError(f"deliberate failure for {symbol}")
            ts = datetime(start.year, start.month, start.day, 14, 30, tzinfo=UTC)
            return pl.DataFrame(
                {
                    "timestamp": pl.Series(
                        [ts], dtype=pl.Datetime(time_unit="us", time_zone="UTC")
                    ),
                    "symbol": pl.Series([symbol], dtype=pl.Utf8),
                    "instrument_id": pl.Series([11667], dtype=pl.UInt32),
                    "open": pl.Series(["1.0"], dtype=pl.Decimal(38, 8)),
                    "high": pl.Series(["1.0"], dtype=pl.Decimal(38, 8)),
                    "low": pl.Series(["1.0"], dtype=pl.Decimal(38, 8)),
                    "close": pl.Series(["1.0"], dtype=pl.Decimal(38, 8)),
                    "volume": pl.Series(["0.0"], dtype=pl.Decimal(38, 2)),
                }
            )
        finally:
            with self.concurrent_lock:
                self.in_flight -= 1


class _ProbeRateLimiter:
    """Records whether ``acquire`` is entered concurrently."""

    instances: list[_ProbeRateLimiter] = []

    def __init__(
        self,
        requests_per_minute: int | None = None,
        burst: int | None = None,
        min_interval_seconds: float | None = None,
    ) -> None:
        self.in_flight = 0
        self.peak_concurrent = 0
        self.lock = threading.Lock()
        _ProbeRateLimiter.instances.append(self)

    def acquire(self) -> None:
        with self.lock:
            self.in_flight += 1
            self.peak_concurrent = max(self.peak_concurrent, self.in_flight)
        try:
            time.sleep(0.01)
        finally:
            with self.lock:
                self.in_flight -= 1


@pytest.fixture
def universe() -> UniverseDefinition:
    return UniverseDefinition(
        name="cohort-test",
        version=1,
        kind=UniverseKind.EXPLICIT,
        spec={"symbols": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]},
    )


@pytest.fixture
def small_universe() -> UniverseDefinition:
    return UniverseDefinition(
        name="cohort-small",
        version=1,
        kind=UniverseKind.EXPLICIT,
        spec={"symbols": ["AAA", "BBB"]},
    )


@pytest.fixture(autouse=True)
def _disable_provider_policy_delays() -> Iterator[None]:
    with patch.dict("liq.data.service.POLICIES", {"databento": None}):
        yield


# ----- argument validation -------------------------------------------------


class TestMaxWorkersValidation:
    def test_default_is_one(self, tmp_path: Path, small_universe: UniverseDefinition) -> None:
        provider = _RecordingProvider(hold_seconds=0.0)
        service = DataService(data_root=tmp_path)
        with patch.object(service, "_get_provider", return_value=provider):
            service.sync(
                small_universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        # Peak concurrent <= 1 in default mode (sequential)
        assert provider.peak_concurrent == 1

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_non_positive_max_workers_rejected(
        self, tmp_path: Path, small_universe: UniverseDefinition, bad_value: int
    ) -> None:
        service = DataService(data_root=tmp_path)
        with pytest.raises(ValueError, match="max_workers"):
            service.sync(
                small_universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=bad_value,
            )


# ----- backwards-compat (default sequential) -------------------------------


class TestSequentialEquivalence:
    def test_max_workers_1_is_sequential(
        self, tmp_path: Path, universe: UniverseDefinition
    ) -> None:
        provider = _RecordingProvider(hold_seconds=0.05)
        service = DataService(data_root=tmp_path)
        with patch.object(service, "_get_provider", return_value=provider):
            service.sync(
                universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=1,
            )
        assert provider.peak_concurrent == 1
        # All 6 symbols fetched
        symbols_called = {c["symbol"] for c in provider.calls}
        assert symbols_called == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}

    def test_default_sync_preserves_existing_return_shape(
        self, tmp_path: Path, small_universe: UniverseDefinition
    ) -> None:
        provider = _RecordingProvider(hold_seconds=0.0)
        service = DataService(data_root=tmp_path)
        with patch.object(service, "_get_provider", return_value=provider):
            result = service.sync(
                small_universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
            )
        for key in ("symbols", "api_calls", "manifest_gaps", "rows_fetched", "pit", "sync_run_id"):
            assert key in result


# ----- bounded concurrency at max_workers > 1 ------------------------------


class TestBoundedConcurrency:
    def test_peak_concurrent_le_max_workers(
        self, tmp_path: Path, universe: UniverseDefinition
    ) -> None:
        provider = _RecordingProvider(hold_seconds=0.05)
        service = DataService(data_root=tmp_path)
        with patch.object(service, "_get_provider", return_value=provider):
            service.sync(
                universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=3,
            )
        assert 1 < provider.peak_concurrent <= 3, (
            f"expected 2-3 concurrent in flight, got {provider.peak_concurrent}"
        )

    def test_all_symbols_complete_in_parallel_mode(
        self, tmp_path: Path, universe: UniverseDefinition
    ) -> None:
        provider = _RecordingProvider(hold_seconds=0.01)
        service = DataService(data_root=tmp_path)
        with patch.object(service, "_get_provider", return_value=provider):
            result = service.sync(
                universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=4,
            )
        symbols_called = {c["symbol"] for c in provider.calls}
        assert symbols_called == {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF"}
        assert result["symbols"] == 6

    def test_aggregated_counters_match_sequential(
        self, tmp_path: Path, small_universe: UniverseDefinition
    ) -> None:
        """Per-symbol row counts accumulate atomically; parallel-mode totals
        match the sequential-mode totals for the same workload."""
        provider_seq = _RecordingProvider(hold_seconds=0.0)
        service = DataService(data_root=tmp_path)
        with patch.object(service, "_get_provider", return_value=provider_seq):
            seq_result = service.sync(
                small_universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=1,
            )

        # Fresh data root so the parallel run sees the same fetch plan.
        tmp_parallel = tmp_path / "parallel"
        tmp_parallel.mkdir()
        provider_par = _RecordingProvider(hold_seconds=0.0)
        service_par = DataService(data_root=tmp_parallel)
        with patch.object(service_par, "_get_provider", return_value=provider_par):
            par_result = service_par.sync(
                small_universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=4,
            )
        assert seq_result["api_calls"] == par_result["api_calls"]
        assert seq_result["rows_fetched"] == par_result["rows_fetched"]
        assert seq_result["manifest_gaps"] == par_result["manifest_gaps"]
        assert seq_result["symbols"] == par_result["symbols"]

    def test_rate_limiter_acquire_is_serialized_in_parallel_mode(
        self, tmp_path: Path, universe: UniverseDefinition
    ) -> None:
        _ProbeRateLimiter.instances.clear()
        provider = _RecordingProvider(hold_seconds=0.01)
        service = DataService(data_root=tmp_path)
        with (
            patch.object(service, "_get_provider", return_value=provider),
            patch("liq.data.service.RateLimiter", _ProbeRateLimiter),
        ):
            service.sync(
                universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=4,
            )
        assert provider.peak_concurrent > 1
        assert _ProbeRateLimiter.instances
        assert _ProbeRateLimiter.instances[0].peak_concurrent == 1


# ----- error propagation ---------------------------------------------------


class TestErrorPropagation:
    def test_per_symbol_failure_propagates(
        self, tmp_path: Path, small_universe: UniverseDefinition
    ) -> None:
        provider = _RecordingProvider(hold_seconds=0.0, failing_symbols=["BBB"])
        service = DataService(data_root=tmp_path)
        with (
            patch.object(service, "_get_provider", return_value=provider),
            pytest.raises(RuntimeError, match="deliberate failure for BBB"),
        ):
            service.sync(
                small_universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=2,
            )

    def test_parallel_failure_rolls_back_failed_symbol_manifest(
        self, tmp_path: Path, universe: UniverseDefinition
    ) -> None:
        provider = _RecordingProvider(hold_seconds=0.0, failing_symbols=["CCC"])
        service = DataService(data_root=tmp_path)
        with (
            patch.object(service, "_get_provider", return_value=provider),
            pytest.raises(RuntimeError, match="deliberate failure for CCC"),
        ):
            service.sync(
                universe,
                start=date(2024, 6, 3),
                end=date(2024, 6, 4),
                provider="databento",
                timeframe="1m",
                dataset="EQUS.MINI",
                max_workers=2,
            )

        aaa = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="AAA",
        )
        ccc = CoverageManifest.load(
            root=tmp_path,
            provider="databento",
            dataset="EQUS.MINI",
            timeframe="1m",
            symbol="CCC",
        )
        assert len(aaa.ranges) == 1
        assert ccc.ranges == []


# ----- file lock semantics -------------------------------------------------


class TestLockSemantics:
    def test_existing_universe_lock_blocks_parallel_sync(
        self, tmp_path: Path, small_universe: UniverseDefinition
    ) -> None:
        lock_path = tmp_path / "locks" / "sync" / "databento--EQUS.MINI--1m--cohort-small.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        held = FileLock(str(lock_path))
        held.acquire()
        try:
            provider = _RecordingProvider(hold_seconds=0.0)
            service = DataService(data_root=tmp_path)
            with (
                patch.object(service, "_get_provider", return_value=provider),
                pytest.raises(SyncLockedError),
            ):
                service.sync(
                    small_universe,
                    start=date(2024, 6, 3),
                    end=date(2024, 6, 4),
                    provider="databento",
                    timeframe="1m",
                    dataset="EQUS.MINI",
                    lock_timeout=0.1,
                    max_workers=4,
                )
        finally:
            held.release()

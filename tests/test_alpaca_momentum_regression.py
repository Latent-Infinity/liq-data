"""Empty-source controls for mandatory authorization and numerical guard wiring."""

import json
import os
import signal

import pytest
from momentum_numerical_test_support import metadata_scope

from liq.data.alpaca_momentum_numerical_models import MomentumNumericalRefused
from liq.data.alpaca_momentum_numerical_scope import adopt_numerical_scope
from liq.data.alpaca_momentum_regression import read_momentum_case
from liq.data.lockbox import CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1, LockboxGuard


def test_usage_fifo_refuses_without_blocking_or_opening_source(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    log = case.usage_root / "numerical-lockbox-usage.jsonl"
    os.mkfifo(log, 0o600)
    guard = LockboxGuard(log, CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1)
    previous = signal.signal(
        signal.SIGALRM, lambda *_: pytest.fail("FIFO blocked before validation")
    )
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    try:
        with (
            adopt_numerical_scope(
                case.contract, case.authority, case.expected(), "reference"
            ) as binding,
            pytest.raises(MomentumNumericalRefused, match="usage_log"),
        ):
            read_momentum_case(binding, guard)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert not (case.usage_root / "reference-daily-opened.json").exists()


def test_default_guard_refuses_before_reservation_or_source_open(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    with (
        adopt_numerical_scope(
            case.contract, case.authority, case.expected(), "reference"
        ) as binding,
        pytest.raises(MomentumNumericalRefused, match="numerical_ledger"),
    ):
        read_momentum_case(binding, LockboxGuard(case.usage_root / "default-log"))
    assert not tuple(case.source_root.iterdir())
    assert not tuple(case.usage_root.iterdir())


def test_wrong_log_path_cannot_reuse_acquisition_log(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    guard = LockboxGuard(
        case.source_root / "old-acquisition-log", CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1
    )
    with (
        adopt_numerical_scope(
            case.contract, case.authority, case.expected(), "reference"
        ) as binding,
        pytest.raises(MomentumNumericalRefused, match="numerical_log_path"),
    ):
        read_momentum_case(binding, guard)
    assert not tuple(case.source_root.iterdir())
    assert not tuple(case.usage_root.iterdir())


def test_declared_empty_case_logs_before_missing_source_and_stops(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    log = case.usage_root / "numerical-lockbox-usage.jsonl"
    guard = LockboxGuard(log, CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1)
    with (
        adopt_numerical_scope(
            case.contract, case.authority, case.expected(), "reference"
        ) as binding,
        pytest.raises(MomentumNumericalRefused, match="file_open"),
    ):
        read_momentum_case(binding, guard)
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["dataset"] == "alpaca_ranker_momentum_numerical_fixture"
    assert entries[0]["purpose"] == "dev_smoke"
    assert entries[0]["start"] == "2022-04-01"
    assert entries[0]["end"] == "2023-03-31"
    assert entries[0]["final_portfolio_review"] is False
    assert not (case.usage_root / "reference-daily-opened.json").exists()
    assert not tuple(case.source_root.iterdir())

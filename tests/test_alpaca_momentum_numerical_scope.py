"""Source-free boundaries for the one retained momentum numerical case."""

from datetime import date

import pytest
from pydantic import ValidationError

from liq.data.alpaca_momentum_numerical_scope import (
    MomentumNumericalRead,
    MomentumNumericalRefused,
    check_numerical_guard,
)
from liq.data.lockbox import (
    CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1,
    CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1,
    INTRADAY_CAMPAIGN_LEDGER_V1,
    LockboxGuard,
)


@pytest.mark.parametrize(
    "changed",
    [
        {"start": date(2022, 3, 31)},
        {"end": date(2023, 4, 1)},
        {"purpose": "characterization"},
        {"arm_id": "classical_cross_sectional_ranker"},
        {"final_portfolio_review": True},
        {"decoded_daily_records": 12},
        {"selected_endpoints": 11},
        {"symbol": "SPY"},
        {"max_numerical_reads": 4},
    ],
)
def test_numerical_case_refuses_scope_variants(changed) -> None:
    with pytest.raises(ValidationError):
        MomentumNumericalRead(**changed)


def test_numerical_read_is_immutable_and_rejects_unknown_fields() -> None:
    request = MomentumNumericalRead()
    with pytest.raises(ValidationError):
        request.purpose = "characterization"
    with pytest.raises(ValidationError):
        MomentumNumericalRead(extra_price_window=True)


@pytest.mark.parametrize(
    "ledger", [INTRADAY_CAMPAIGN_LEDGER_V1, CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1]
)
def test_numerical_guard_rejects_default_or_consumed_acquisition_ledger(tmp_path, ledger) -> None:
    log = tmp_path / "usage.jsonl"
    guard = LockboxGuard(log, ledger)
    with pytest.raises(MomentumNumericalRefused, match="numerical_ledger"):
        check_numerical_guard(MomentumNumericalRead(), guard)
    assert not log.exists()


def test_numerical_guard_revalidates_copied_request_before_log(tmp_path) -> None:
    log = tmp_path / "usage.jsonl"
    guard = LockboxGuard(log, CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1)
    request = MomentumNumericalRead().model_copy(update={"end": date(2024, 1, 1)})
    with pytest.raises(MomentumNumericalRefused, match="numerical_request"):
        check_numerical_guard(request, guard)
    assert not log.exists()


def test_numerical_ledger_is_distinct_and_does_not_widen_acquisition() -> None:
    ledger = CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1
    assert ledger is not CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1
    assert set(ledger.datasets) == {"alpaca_ranker_momentum_numerical_fixture"}
    windows = ledger.datasets["alpaca_ranker_momentum_numerical_fixture"]
    assert windows.characterization == (date(2022, 4, 1), date(2023, 3, 31))
    assert windows.allowed_purposes == frozenset({"dev_smoke"})
    assert all(
        item.allowed_purposes == frozenset({"characterization"})
        for item in CLASSICAL_RANKER_MOMENTUM_FIXTURE_LEDGER_V1.datasets.values()
    )

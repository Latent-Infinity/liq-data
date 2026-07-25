"""Parser + model tests against real committed EDGAR companyfacts fixtures.

Golden values are real (SEC EDGAR, fetched 2026-07-22). Derivation None-paths
use directly-constructed models (logic tests, not fabricated market data).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from liq.data.fundamentals.concepts import CONCEPT_CANDIDATES, unit_for
from liq.data.fundamentals.models import AnnualFundamentals
from liq.data.fundamentals.parser import build_snapshot, resolve_concept

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "edgar"
AS_OF = date(2026, 3, 31)


def _facts(symbol: str) -> dict:
    return json.loads((FIXTURES / f"{symbol}.companyfacts.json").read_text())


def _snap(symbol: str, as_of: date = AS_OF):
    cf = _facts(symbol)
    return build_snapshot(symbol, str(cf["cik"]).zfill(10), cf, as_of)


class TestGoldenDerivations:
    def test_msft(self) -> None:
        a = _snap("MSFT").latest
        assert a is not None
        assert a.fiscal_year == 2025
        assert a.fiscal_year_end == date(2025, 6, 30)
        assert a.revenue == pytest.approx(281.7e9, rel=0.01)
        assert a.fcf == pytest.approx(71.6e9, rel=0.02)
        assert a.ebitda == pytest.approx(150.5e9, rel=0.02)
        assert a.net_debt == pytest.approx(12.9e9, rel=0.05)
        assert a.interest_coverage == pytest.approx(53.9, rel=0.02)

    def test_nvda_net_cash(self) -> None:
        a = _snap("NVDA").latest
        assert a is not None
        assert a.fiscal_year == 2026
        assert a.fcf == pytest.approx(96.7e9, rel=0.02)
        assert a.net_debt is not None and a.net_debt < 0  # net cash
        assert a.interest_coverage is not None and a.interest_coverage > 100

    def test_txn_leverage(self) -> None:
        a = _snap("TXN").latest
        assert a is not None
        assert a.fiscal_year == 2025
        assert a.net_debt is not None and a.ebitda is not None
        assert a.net_debt / a.ebitda == pytest.approx(1.36, rel=0.03)
        assert a.interest_coverage == pytest.approx(11.1, rel=0.03)

    def test_all_have_five_positive_fcf_years(self) -> None:
        for sym in ("MSFT", "NVDA", "TXN"):
            positive, considered = _snap(sym).fcf_positive_count(5)
            assert considered == 5
            assert positive == 5


class TestTagResolution:
    def test_revenue_tag_differs_by_filer(self) -> None:
        # NVDA's recent total revenue lives under `Revenues`; MSFT/TXN under the
        # contract-revenue tag. Latest-coverage-wins must pick correctly.
        for sym, expected in (
            ("NVDA", "Revenues"),
            ("MSFT", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ):
            gaap = _facts(sym)["facts"]["us-gaap"]
            tag, _ = resolve_concept(
                gaap, CONCEPT_CANDIDATES["revenue"], unit_for("revenue"), AS_OF
            )
            assert tag == expected

    def test_interest_migrates_to_nonoperating(self) -> None:
        gaap = _facts("MSFT")["facts"]["us-gaap"]
        tag, _ = resolve_concept(
            gaap, CONCEPT_CANDIDATES["interest_expense"], unit_for("interest_expense"), AS_OF
        )
        assert tag == "InterestExpenseNonoperating"


class TestPointInTime:
    def test_future_filing_excluded(self) -> None:
        # MSFT's FY2025 10-K was filed 2025-07-30; before that, FY2024 is latest.
        early = _snap("MSFT", as_of=date(2025, 1, 1)).latest
        assert early is not None
        assert early.fiscal_year == 2024

    def test_series_is_ascending_and_pit_bounded(self) -> None:
        snap = _snap("TXN")
        years = [a.fiscal_year for a in snap.annual]
        assert years == sorted(years)
        assert all(a.filed <= AS_OF for a in snap.annual)


class TestDerivationNonePaths:
    def _row(self, **kw) -> AnnualFundamentals:
        base = {
            "fiscal_year": 2025,
            "fiscal_year_end": date(2025, 12, 31),
            "filed": date(2026, 2, 1),
        }
        return AnnualFundamentals(**{**base, **kw})

    def test_fcf_none_without_capex(self) -> None:
        assert self._row(cfo=100.0).fcf is None

    def test_ebitda_none_without_dep_amort(self) -> None:
        assert self._row(operating_income=50.0).ebitda is None

    def test_net_debt_none_without_cash(self) -> None:
        assert self._row(lt_debt_noncurrent=10.0).net_debt is None

    def test_total_debt_absent_current_treated_as_present_noncurrent(self) -> None:
        assert self._row(lt_debt_noncurrent=10.0, cash=3.0).net_debt == pytest.approx(7.0)

    def test_coverage_none_on_nonpositive_interest(self) -> None:
        assert self._row(operating_income=50.0, interest_expense=0.0).interest_coverage is None

    def test_sbc_ratio_none_on_nonpositive_fcf(self) -> None:
        assert self._row(sbc=5.0, cfo=1.0, capex=2.0).sbc_to_fcf is None

    def test_total_debt_none_without_any_debt(self) -> None:
        row = self._row(cash=5.0)
        assert row.total_debt is None
        assert row.net_debt is None


class TestEmptySnapshot:
    def test_empty_snapshot_degrades_gracefully(self) -> None:
        from liq.data.fundamentals.models import FundamentalsSnapshot

        snap = FundamentalsSnapshot(symbol="X", cik="0000000000", as_of=date(2026, 1, 1), annual=())
        assert snap.latest is None
        assert snap.missing_latest_concepts() == ()
        assert snap.fcf_positive_count() == (0, 0)
        assert snap.diluted_share_growth() is None

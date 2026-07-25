"""Canonical fundamentals concepts and their XBRL tag-priority candidates.

Filers tag the same economic quantity with different ``us-gaap`` concepts,
and the same filer switches tags across eras. These candidate lists were
derived from **real** EDGAR companyfacts for a diversified set of filers
(a hyperscaler/software name, an AI-accelerator, a diversified semi), and
resolution is *latest-coverage-wins* (the candidate whose annual 10-K facts
reach the most recent fiscal year), not first-present — because e.g. one
filer's recent revenue lives under ``Revenues`` while its older years sit
under ``RevenueFromContractWithCustomerExcludingAssessedTax``.
"""

from __future__ import annotations

# Canonical concept -> ordered candidate ``us-gaap`` tags (priority breaks ties
# when two candidates reach the same latest fiscal year).
CONCEPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "dep_amort": (
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ),
    "cfo": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "lt_debt_noncurrent": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "lt_debt_current": ("LongTermDebtCurrent", "DebtCurrent"),
    "interest_expense": (
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
    ),
    "diluted_shares": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "sbc": ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
    "inventory": ("InventoryNet",),
}

# Concepts reported in share units rather than USD.
SHARE_CONCEPTS: frozenset[str] = frozenset({"diluted_shares"})


def unit_for(concept: str) -> str:
    """Return the XBRL unit key a concept is reported in."""
    return "shares" if concept in SHARE_CONCEPTS else "USD"


__all__ = ["CONCEPT_CANDIDATES", "SHARE_CONCEPTS", "unit_for"]

"""Point-in-time fundamentals domain models with derived metrics.

``AnnualFundamentals`` holds one fiscal year's raw concept values (each
``None`` when the filer did not tag it — never imputed) and exposes the
derived quantities the ASD-25 fundamental gates need. Derivations return
``None`` when a required input is absent, so the gate layer can mark a check
``UNSCORED`` rather than fabricate a number.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict


class CommonSharesFact(BaseModel):
    """One accession-bound common-shares fact known by a PIT cutoff."""

    model_config = ConfigDict(frozen=True)

    value: float
    period_end: date
    filed: date
    accession_number: str
    form: str


class AnnualFundamentals(BaseModel):
    """One fiscal year of annual (10-K) fundamentals, as known at a filing."""

    model_config = ConfigDict(frozen=True)

    fiscal_year: int
    fiscal_year_end: date
    filed: date

    revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    dep_amort: float | None = None
    cfo: float | None = None
    capex: float | None = None
    cash: float | None = None
    lt_debt_current: float | None = None
    lt_debt_noncurrent: float | None = None
    interest_expense: float | None = None
    diluted_shares: float | None = None
    sbc: float | None = None
    inventory: float | None = None

    @property
    def fcf(self) -> float | None:
        """Free cash flow = operating cash flow − capex."""
        if self.cfo is None or self.capex is None:
            return None
        return self.cfo - self.capex

    @property
    def ebitda(self) -> float | None:
        """Approximate EBITDA = operating income + D&A.

        ``None`` when D&A is not tagged (some filers do not report a combined
        depreciation-and-amortization line); the net-debt/EBITDA gate is then
        ``UNSCORED`` rather than imputed. Where only ``Depreciation`` is
        available, EBITDA is a slight understatement (a conservative bias for
        a leverage ratio).
        """
        if self.operating_income is None or self.dep_amort is None:
            return None
        return self.operating_income + self.dep_amort

    @property
    def total_debt(self) -> float | None:
        """Interest-bearing debt = current + non-current portions.

        A missing current portion is treated as zero **only if** the
        non-current portion is present (firms with no near-term maturities
        simply omit the line); if neither debt component is tagged, ``None``.
        """
        parts = [d for d in (self.lt_debt_current, self.lt_debt_noncurrent) if d is not None]
        return sum(parts) if parts else None

    @property
    def net_debt(self) -> float | None:
        """Net debt = total debt − cash. ``None`` if either is unavailable."""
        if self.total_debt is None or self.cash is None:
            return None
        return self.total_debt - self.cash

    @property
    def interest_coverage(self) -> float | None:
        """Operating income / interest expense. ``None`` if not computable.

        Zero or negative reported interest expense yields ``None`` (coverage
        is undefined / trivially unbounded and should not pass as a number).
        """
        if self.operating_income is None or self.interest_expense is None:
            return None
        if self.interest_expense <= 0.0:
            return None
        return self.operating_income / self.interest_expense

    @property
    def sbc_to_fcf(self) -> float | None:
        """Stock-based comp as a fraction of FCF. ``None`` if FCF ≤ 0/absent."""
        fcf = self.fcf
        if self.sbc is None or fcf is None or fcf <= 0.0:
            return None
        return self.sbc / fcf


class FundamentalsSnapshot(BaseModel):
    """As-of-``T`` view: annual fundamentals with ``filed <= as_of`` only."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    cik: str
    as_of: date
    annual: tuple[AnnualFundamentals, ...]  # ascending by fiscal year

    @property
    def latest(self) -> AnnualFundamentals | None:
        """Most recent fiscal year known as of ``as_of``."""
        return self.annual[-1] if self.annual else None

    def fcf_positive_count(self, years: int = 5) -> tuple[int, int]:
        """Return ``(positive_years, considered_years)`` over the last ``years``.

        Only fiscal years with a computable FCF count toward the denominator,
        so a filer missing cash-flow tags is not silently scored as failing.
        """
        recent = self.annual[-years:]
        fcfs = [a.fcf for a in recent if a.fcf is not None]
        positive = sum(1 for f in fcfs if f > 0.0)
        return positive, len(fcfs)

    def diluted_share_growth(self) -> float | None:
        """Latest-year diluted share count growth vs the prior year."""
        counts = [a.diluted_shares for a in self.annual if a.diluted_shares is not None]
        if len(counts) < 2 or counts[-2] <= 0.0:
            return None
        return counts[-1] / counts[-2] - 1.0

    def inventory_growth_vs_sales(self) -> float | None:
        """Inventory growth minus revenue growth over the last two fiscal years.

        Positive means inventory is building faster than sales — a channel /
        demand-softening risk. ``None`` if either year lacks inventory or
        revenue (many software names carry no inventory: legitimately absent).
        """
        if len(self.annual) < 2:
            return None
        prior, latest = self.annual[-2], self.annual[-1]
        if None in (prior.inventory, latest.inventory, prior.revenue, latest.revenue):
            return None
        if prior.inventory <= 0.0 or prior.revenue <= 0.0:  # type: ignore[operator]
            return None
        inv_growth = latest.inventory / prior.inventory - 1.0  # type: ignore[operator]
        sales_growth = latest.revenue / prior.revenue - 1.0  # type: ignore[operator]
        return inv_growth - sales_growth

    def missing_latest_concepts(self) -> tuple[str, ...]:
        """Concepts absent in the latest fiscal year (for the exclusion report)."""
        latest = self.latest
        if latest is None:
            return ()
        fields = (
            "revenue",
            "operating_income",
            "net_income",
            "dep_amort",
            "cfo",
            "capex",
            "cash",
            "lt_debt_noncurrent",
            "interest_expense",
            "diluted_shares",
            "sbc",
        )
        return tuple(f for f in fields if getattr(latest, f) is None)


__all__ = ["AnnualFundamentals", "CommonSharesFact", "FundamentalsSnapshot"]

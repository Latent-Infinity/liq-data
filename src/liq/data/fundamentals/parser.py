"""Pure normalization of EDGAR companyfacts into PIT annual fundamentals.

No network here — these functions take an already-fetched ``companyfacts``
dict and produce a :class:`FundamentalsSnapshot`. Selection is
point-in-time: only annual (10-K, ``fp="FY"``) facts **filed on or before**
``as_of`` are used, and per fiscal year the latest such filing wins
(restatement-aware). Concept resolution is latest-coverage-wins.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from liq.data.fundamentals.concepts import CONCEPT_CANDIDATES, unit_for
from liq.data.fundamentals.models import AnnualFundamentals, FundamentalsSnapshot

_Fact = dict[str, Any]


def _annual_facts(node: _Fact, unit: str, filed_le: date) -> dict[str, _Fact]:
    """Annual 10-K facts for one concept node, keyed by **period-end date**.

    Keyed by ``end`` (not ``fy``): EDGAR's ``fy``/``fp`` describe the filing,
    so a 10-K's two comparative years share one ``fy`` — keying by fiscal year
    would collapse them. Keeps ``form`` starting ``10-K`` with ``fp="FY"``,
    ``filed <= filed_le``, and — for flow concepts (those with a ``start``) —
    a period length of ~1 year, excluding any YTD/partial periods. Per end
    date the latest-filed fact wins (restatement-aware within the PIT cutoff).
    """
    out: dict[str, _Fact] = {}
    for entry in node.get("units", {}).get(unit, []):
        if not str(entry.get("form", "")).startswith("10-K") or entry.get("fp") != "FY":
            continue
        filed = entry.get("filed")
        end = entry.get("end")
        if filed is None or end is None or entry.get("val") is None:
            continue
        if date.fromisoformat(filed) > filed_le:
            continue
        start = entry.get("start")
        if start is not None:
            span = (date.fromisoformat(end) - date.fromisoformat(start)).days
            if not 330 <= span <= 400:  # annual periods only
                continue
        prev = out.get(end)
        if prev is None or filed > prev["filed"]:
            out[end] = entry
    return out


def resolve_concept(
    gaap: dict[str, _Fact],
    candidates: tuple[str, ...],
    unit: str,
    filed_le: date,
) -> tuple[str | None, dict[str, _Fact]]:
    """Pick the candidate tag with the latest annual coverage (priority breaks ties)."""
    best_tag: str | None = None
    best_facts: dict[str, _Fact] = {}
    best_key = ("", 1)
    for index, tag in enumerate(candidates):
        node = gaap.get(tag)
        if not node:
            continue
        facts = _annual_facts(node, unit, filed_le)
        if not facts:
            continue
        key = (max(facts), -index)  # latest period end wins; earlier candidate breaks ties
        if key > best_key:
            best_key, best_tag, best_facts = key, tag, facts
    return best_tag, best_facts


def build_snapshot(
    symbol: str,
    cik: str,
    companyfacts: dict[str, Any],
    as_of: date,
    *,
    max_years: int = 6,
) -> FundamentalsSnapshot:
    """Normalize ``companyfacts`` into an as-of-``T`` :class:`FundamentalsSnapshot`."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    resolved: dict[str, dict[str, _Fact]] = {
        concept: resolve_concept(gaap, candidates, unit_for(concept), as_of)[1]
        for concept, candidates in CONCEPT_CANDIDATES.items()
    }

    # Align concepts on their shared fiscal-year-end date (income-statement
    # period end == balance-sheet instant for a 10-K).
    ends = sorted({end for facts in resolved.values() for end in facts})[-max_years:]
    annual: list[AnnualFundamentals] = []
    for end in ends:
        present = {concept: facts[end] for concept, facts in resolved.items() if end in facts}
        if not present:
            continue
        fiscal_year_end = date.fromisoformat(end)
        annual.append(
            AnnualFundamentals(
                fiscal_year=fiscal_year_end.year,
                fiscal_year_end=fiscal_year_end,
                filed=max(date.fromisoformat(f["filed"]) for f in present.values()),
                **{concept: float(fact["val"]) for concept, fact in present.items()},
            )
        )
    return FundamentalsSnapshot(symbol=symbol, cik=cik, as_of=as_of, annual=tuple(annual))


__all__ = ["build_snapshot", "resolve_concept"]

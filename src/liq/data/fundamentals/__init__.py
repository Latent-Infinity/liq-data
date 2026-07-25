"""Fundamentals normalization: EDGAR XBRL companyfacts → PIT annual metrics.

The concept map and parser are pure (no network); the fetching provider is
``liq.data.providers.sec_edgar_fundamentals``.
"""

from liq.data.fundamentals.concepts import CONCEPT_CANDIDATES, SHARE_CONCEPTS, unit_for
from liq.data.fundamentals.models import AnnualFundamentals, FundamentalsSnapshot
from liq.data.fundamentals.parser import build_snapshot, resolve_concept

__all__ = [
    "CONCEPT_CANDIDATES",
    "SHARE_CONCEPTS",
    "AnnualFundamentals",
    "FundamentalsSnapshot",
    "build_snapshot",
    "resolve_concept",
    "unit_for",
]

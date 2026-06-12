"""Phase 0 contract stub: ``DatabentoProvider`` exists (Phase 1 deliverable).

Strict xfail flips green when Phase 1 lands the provider per
liq-scan-plan §3.2. The full provider contract (batch routing, dataset
mapping, symbology persistence, normalization) gets exercised in
Phase 1's real tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="Phase 1 deliverable — DatabentoProvider not yet implemented",
)
def test_databento_provider_importable() -> None:
    from liq.data.providers.databento import DatabentoProvider  # noqa: PLC0415

    assert DatabentoProvider is not None

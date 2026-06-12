"""Phase 0 contract stub: coverage manifest exists (Phase 2 deliverable).

Strict xfail flips green when Phase 2 lands the manifest per
liq-scan-plan §3.4 / plan §7.3. The manifest is the idempotency
mechanism for sync(universe); its full surface (gap detection, merge,
transactional rollback) gets exercised in Phase 2's real tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 deliverable — coverage manifest not yet implemented",
)
def test_coverage_manifest_importable() -> None:
    from liq.data.manifest import CoverageManifest  # noqa: PLC0415

    assert CoverageManifest is not None

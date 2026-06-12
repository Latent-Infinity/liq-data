"""Contract stub for the coverage manifest (planned, not built).

Strict xfail. Flips green when the manifest lands per the liq-scan
plan §3.4 / §7.3. The manifest is the idempotency mechanism for
``sync(universe)``; its full surface (gap detection, merge, transactional
rollback) gets exercised by the real implementation's tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="coverage manifest not yet implemented (planned)",
)
def test_coverage_manifest_importable() -> None:
    from liq.data.manifest import CoverageManifest  # noqa: PLC0415

    assert CoverageManifest is not None

"""Phase 0 contract stub: ``UniverseDefinition`` exists (Phase 2 deliverable).

Strict xfail flips green when Phase 2 lands the universe machinery per
liq-scan-plan §3.4. The full surface — four kinds (explicit / filter /
composite / set_op), PIT flag propagation, set algebra — gets exercised
in Phase 2's real tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="Phase 2 deliverable — UniverseDefinition not yet implemented",
)
def test_universe_definition_importable() -> None:
    from liq.data.universes import UniverseDefinition  # noqa: PLC0415

    assert UniverseDefinition is not None

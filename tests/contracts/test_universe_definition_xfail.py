"""Contract stub for ``UniverseDefinition`` (planned, not built).

Strict xfail. Flips green when the universe machinery lands per the
liq-scan plan §3.4. The full surface — four kinds (explicit / filter /
composite / set_op), PIT flag propagation, set algebra — gets exercised
by the real implementation's tests.
"""

from __future__ import annotations

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="UniverseDefinition not yet implemented (planned)",
)
def test_universe_definition_importable() -> None:
    from liq.data.universes import UniverseDefinition  # noqa: PLC0415

    assert UniverseDefinition is not None

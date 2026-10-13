"""Bounded reader for one Oracle-admitted, retained engineering fixture.

No transport or historical admission is provided. The caller must freeze code
and provenance before access under the October 3 ranker implementation decision.
"""

import hashlib
import json
import os
import stat
from decimal import Decimal
from pathlib import Path

LABELS = ("2023-04-21T13:30:00Z", "2023-04-21T13:31:00Z")
PAGES = (
    (
        "KHC",
        "bars-khc-raw/page-1.bin",
        80520,
        "22bc1b04f5341c22dc92fd9ee0ebe121e94ad9961bbad81fbe95b034e39e17e0",
    ),
    (
        "SPY",
        "bars-spy-raw/page-1.bin",
        106850,
        "a86e6fa4dc26b08d70486cf4666b5b589e0af245530c1181bf00c3e5ccdab49c",
    ),
)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate fixture JSON key")
        result[key] = value
    return result


def read_return_fixture(root: Path) -> dict[str, tuple[Decimal, Decimal]]:
    """Extract only the four pinned raw close fields, refusing any discrepancy."""
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("fixture path contains symlink")
    if not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError("fixture root must be private (0700)")
    result: dict[str, tuple[Decimal, Decimal]] = {}
    for symbol, relative, size, digest in PAGES:
        path = root / relative
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("fixture page contains symlink")
        if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
            raise ValueError("fixture page directory must be private (0700)")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ValueError("fixture page must be a private regular file (0600)")
            if metadata.st_size != size:
                raise ValueError("fixture page identity mismatch")
            raw = source.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("fixture page identity mismatch")
        payload = json.loads(raw, parse_float=Decimal, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict):
            raise ValueError("fixture bars schema")
        bars = payload["bars"]
        if set(bars) != {symbol} or not isinstance(bars[symbol], list):
            raise ValueError("fixture symbol schema")
        rows = bars[symbol]
        if len(rows) > 1000 or any(not isinstance(row, dict) for row in rows):
            raise ValueError("fixture row budget or schema")
        values: list[Decimal] = []
        for label in LABELS:
            selected = [row for row in rows if row.get("t") == label]
            if len(selected) != 1:
                raise ValueError("fixture exact timestamp missing or duplicated")
            value = selected[0].get("c")
            if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
                raise ValueError("fixture close must be numeric")
            close = Decimal(value)
            if not close.is_finite() or close <= 0:
                raise ValueError("fixture close must be finite positive")
            values.append(close)
        result[symbol] = (values[0], values[1])
    return result

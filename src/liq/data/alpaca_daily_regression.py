"""Read only the independently admitted, private current-vintage daily case."""

import hashlib
import os
import stat
from datetime import datetime
from pathlib import Path

from liq.data.alpaca_daily_fixture import DailyFixture, parse_daily_fixture
from liq.data.alpaca_source import AlpacaDailyFixtureRequest

_WITNESSES = (
    ("D_daily.json", "a2575f2b04ea09e575e58a85f5c759825303f6216833d0340fdf688c209dea84"),
    ("lockbox_usage_log.jsonl", "0cefd64581156872013d2405729b997191a06f8afa984bc289be859daa8a6839"),
    ("KHC-capture.json", "9367548723eddfd809b411c8deb1175ebd91354cb563e67ea1d6c36d0656c460"),
    ("SPY-capture.json", "dfde4973c90613614b8df8f683f5ddc8e326e534397c85eba587eabab7b7d44a"),
    ("KHC-attempt.json", "154d2c6d929fb7d139dd3eb705a634d36dcc56fbe3c61b1312a847cb514d549a"),
    ("SPY-attempt.json", "a97517db9cdc8956c19591faec8c5ca45662305eca9aaf6953e13746780872af"),
)
_PAGES = (
    (
        "KHC",
        "04ef213ee0f7365c0aa67e88b9d1decbe10609639aff3d9baec36e7625e8458b",
        6228,
        "2026-10-03T06:41:16.197226+00:00",
    ),
    (
        "SPY",
        "bdbd71d956e1b57d5d0cbecb259afc9dec8ffc92436459ace8c15a974f52c162",
        6666,
        "2026-10-03T06:41:16.536665+00:00",
    ),
)


def _read(root: Path, name: str, digest: str, limit: int) -> bytes:
    path = root / name
    if path.is_symlink():
        raise ValueError("private daily witness contains symlink")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("private daily witness must be regular and 0600")
        if metadata.st_size >= limit:
            raise ValueError("private daily witness exceeds byte budget")
        raw = source.read(limit)
        if len(raw) >= limit or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("private daily witness identity mismatch")
    return raw


def read_daily_pair(root: Path, labels: tuple[datetime, ...]) -> tuple[DailyFixture, DailyFixture]:
    """No transport, alternatives or historical admission; preserve original clocks."""
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("private daily fixture path contains symlink")
    if not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError("private daily fixture root must exist and be 0700")
    for name, digest in _WITNESSES:
        _read(root, name, digest, 8192)
    fixtures = []
    for symbol, digest, size, clock in _PAGES:
        raw = _read(root, f"{symbol}.json", digest, size + 1)
        if len(raw) != size:
            raise ValueError("private daily page size mismatch")
        fixtures.append(
            parse_daily_fixture(
                raw, AlpacaDailyFixtureRequest(symbol=symbol), labels, datetime.fromisoformat(clock)
            )
        )
    return fixtures[0], fixtures[1]

"""Nonfinancial refusal controls for one hash-pinned private daily fixture."""

import os
from pathlib import Path

import pytest

from liq.data.alpaca_daily_regression import read_daily_pair


def test_missing_private_fixture_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="private"):
        read_daily_pair(tmp_path / "absent", ())


def test_symlink_fixture_root_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        read_daily_pair(link, ())


@pytest.mark.parametrize("kind", ["untrusted", "oversize", "world_readable", "fifo", "symlink"])
def test_untrusted_receipt_refuses_before_price_page_access(tmp_path: Path, kind: str) -> None:
    os.chmod(tmp_path, 0o700)
    path = tmp_path / "D_daily.json"
    if kind == "fifo":
        os.mkfifo(path, 0o600)
    elif kind == "symlink":
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_bytes(b"x" * 8193 if kind == "oversize" else b"{}")
        os.chmod(path, 0o644 if kind == "world_readable" else 0o600)
    with pytest.raises(ValueError):
        read_daily_pair(tmp_path, ())

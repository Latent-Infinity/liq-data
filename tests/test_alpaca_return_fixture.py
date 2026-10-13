"""Private fixture access refuses absent, aliased or altered input."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from liq.data.alpaca_return_fixture import read_return_fixture


def test_fixture_requires_private_directory(tmp_path: Path) -> None:
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        read_return_fixture(tmp_path)


def test_fixture_refuses_symlink(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        read_return_fixture(alias)


def test_fixture_refuses_changed_page(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    directory = tmp_path / "bars-khc-raw"
    directory.mkdir(mode=0o700)
    page = directory / "page-1.bin"
    page.write_bytes(b"")
    page.chmod(0o600)
    with pytest.raises(ValueError, match="identity"):
        read_return_fixture(tmp_path)


def test_fixture_refuses_fifo_without_waiting_for_writer(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    directory = tmp_path / "bars-khc-raw"
    directory.mkdir(mode=0o700)
    os.mkfifo(directory / "page-1.bin", mode=0o600)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from liq.data.alpaca_return_fixture import read_return_fixture; "
            "read_return_fixture(Path(sys.argv[1]))",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode != 0
    assert "private regular file" in result.stderr

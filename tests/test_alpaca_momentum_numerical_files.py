"""Nonfinancial file/identity controls; no market page or signal is fabricated."""

import os
from pathlib import Path
from typing import Literal, assert_never

import pytest

from liq.data.alpaca_momentum_numerical_files import (
    digest,
    hold_file,
    pin_file,
    private_root,
    read_pin,
    write_once,
)
from liq.data.alpaca_momentum_numerical_models import CapturedFile, MomentumNumericalRefused


def make_file(path: Path) -> Path:
    path.write_bytes(b"nonfinancial metadata")
    path.chmod(0o600)
    return path


def test_same_bounded_original_and_readless_fd_are_reconciled(tmp_path) -> None:
    path = make_file(tmp_path / "metadata")
    pin = pin_file(path)
    assert read_pin(pin) == b"nonfinancial metadata"
    with hold_file(path, 100, read_body=False) as original:
        info = path.stat()
        witness = CapturedFile(
            name=path.name,
            sha256=pin.sha256,
            bytes=info.st_size,
            device=info.st_dev,
            inode=info.st_ino,
            mtime_ns=info.st_mtime_ns,
            ctime_ns=info.st_ctime_ns,
        )
        assert original.raw == b""
        original.require_capture(witness)
        with pytest.raises(MomentumNumericalRefused, match="capture_stat"):
            original.require_capture(witness.model_copy(update={"inode": info.st_ino + 1}))


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "size", "mode"])
def test_private_original_rejects_unsafe_file_policy(
    tmp_path, kind: Literal["symlink", "hardlink", "fifo", "size", "mode"]
) -> None:
    path = make_file(tmp_path / "metadata")
    match kind:
        case "symlink":
            path = tmp_path / "alias"
            path.symlink_to(tmp_path / "metadata")
        case "hardlink":
            os.link(path, tmp_path / "alias")
        case "fifo":
            path.unlink()
            os.mkfifo(path)
        case "mode":
            path.chmod(0o644)
        case "size":
            path.write_bytes(b"metadata" * 100)
        case _:
            assert_never(kind)
    with pytest.raises(MomentumNumericalRefused), hold_file(path, 100):
        pytest.fail("unsafe original was adopted")


@pytest.mark.parametrize("kind", ["bytes", "same_bytes_inode", "delete"])
def test_held_original_mutation_refuses_through_context_exit(
    tmp_path, kind: Literal["bytes", "same_bytes_inode", "delete"]
) -> None:
    path = make_file(tmp_path / "metadata")
    with pytest.raises(MomentumNumericalRefused), hold_file(path, 100) as original:
        match kind:
            case "bytes":
                path.write_bytes(b"changed metadata")
            case "same_bytes_inode":
                replacement = make_file(tmp_path / "replacement")
                replacement.replace(path)
            case "delete":
                path.unlink()
            case _:
                assert_never(kind)
        original.check()


def test_pin_digest_mismatch_and_bad_paths_refuse(tmp_path) -> None:
    path = make_file(tmp_path / "metadata")
    pin = pin_file(path)
    with pytest.raises(MomentumNumericalRefused, match="file_pin"):
        read_pin(pin.model_copy(update={"sha256": digest(b"wrong metadata")}))
    with pytest.raises(MomentumNumericalRefused), hold_file(Path("relative"), 100):
        pytest.fail("relative path adopted")
    with pytest.raises(MomentumNumericalRefused), hold_file(path, 0):
        pytest.fail("zero budget adopted")


def test_output_is_exclusive_and_root_policy_is_enforced(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    assert private_root(root) == (root.stat().st_dev, root.stat().st_ino)
    write_once(root, "marker", b"nonfinancial metadata")
    with pytest.raises(MomentumNumericalRefused, match="output_claim"):
        write_once(root, "marker", b"replacement")
    with pytest.raises(MomentumNumericalRefused, match="output_name"):
        write_once(root, "../outside", b"metadata")
    root.chmod(0o755)
    with pytest.raises(MomentumNumericalRefused, match="private_root"):
        private_root(root)

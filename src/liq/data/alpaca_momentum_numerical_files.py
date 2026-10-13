"""Held bounded metadata and readless original-source witnesses."""

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from liq.data.alpaca_momentum_numerical_models import (
    CapturedFile,
    FilePin,
    MomentumNumericalRefused,
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def state(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_mode,
        info.st_uid,
    )


@dataclass(frozen=True, slots=True)
class Original:
    path: Path
    descriptor: int
    identity: tuple[int, ...]
    raw: bytes = field(repr=False)

    def check(self) -> None:
        try:
            if (
                self.path.resolve() != self.path
                or state(self.path.lstat()) != self.identity
                or state(os.fstat(self.descriptor)) != self.identity
            ):
                raise MomentumNumericalRefused("original_changed")
        except OSError as exc:
            raise MomentumNumericalRefused("original_missing") from exc

    def require_capture(self, witness: CapturedFile) -> None:
        self.check()
        if self.identity[:5] != (
            witness.device,
            witness.inode,
            witness.bytes,
            witness.mtime_ns,
            witness.ctime_ns,
        ):
            raise MomentumNumericalRefused("capture_stat")


@contextmanager
def hold_file(
    path: Path, limit: int, *, private: bool = True, read_body: bool = True
) -> Iterator[Original]:
    if path != path.absolute() or path.resolve() != path or limit < 1:
        raise MomentumNumericalRefused("file_path_or_limit")
    if any(parent.is_symlink() for parent in path.parents):
        raise MomentumNumericalRefused("ancestor_symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise MomentumNumericalRefused("file_open") from exc
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise MomentumNumericalRefused("bounded_regular_file")
        if private and (
            info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
            or info.st_uid != os.geteuid()
        ):
            raise MomentumNumericalRefused("private_original")
        raw = stream.read(limit + 1) if read_body else b""
        if len(raw) > limit:
            raise MomentumNumericalRefused("file_byte_budget")
        original = Original(path, stream.fileno(), state(info), raw)
        original.check()
        yield original
        original.check()


def require_pin(original: Original, pin: FilePin) -> None:
    original.check()
    if len(original.raw) != pin.bytes or digest(original.raw) != pin.sha256:
        raise MomentumNumericalRefused("file_pin")


def read_pin(pin: FilePin, limit: int = 536870912) -> bytes:
    with hold_file(Path(pin.path), limit, private=False) as original:
        require_pin(original, pin)
        return original.raw


def pin_file(path: Path) -> FilePin:
    with hold_file(path, 536870912, private=False) as original:
        return FilePin(path=str(path), sha256=digest(original.raw), bytes=len(original.raw))


def private_root(path: Path) -> tuple[int, int]:
    try:
        info = path.lstat()
        if (
            path.resolve() != path
            or any(parent.is_symlink() for parent in path.parents)
            or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid()
        ):
            raise MomentumNumericalRefused("private_root")
        return info.st_dev, info.st_ino
    except OSError as exc:
        raise MomentumNumericalRefused("private_root_missing") from exc


def write_once(root: Path, name: str, raw: bytes) -> tuple[int, ...]:
    before = private_root(root)
    if Path(name).name != name:
        raise MomentumNumericalRefused("output_name")
    try:
        descriptor = os.open(
            root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
    except OSError as exc:
        raise MomentumNumericalRefused("output_claim") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        if private_root(root) != before:
            raise MomentumNumericalRefused("output_root_changed")
        return state(os.fstat(stream.fileno()))

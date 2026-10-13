"""Failure-retained current-instrument audit, separate from historical period guards."""

import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from liq.data.exceptions import ProviderError, ValidationError
from liq.data.oanda_instrument_metadata import (
    MATRIX_SYMBOLS,
    InstrumentMetadataObservation,
    InstrumentMetadataRequest,
)
from liq.data.protocols import InstrumentMetadataProvider
from liq.data.providers.oanda_account_probe import PROBE_ARM

if TYPE_CHECKING:
    from liq.data.service import DataService


@dataclass(frozen=True, slots=True)
class _AuditDirectory:
    path: Path
    descriptor: int


@contextmanager
def _claim_directory(path: Path) -> Iterator[_AuditDirectory]:
    """Pin a no-follow directory chain and sync every newly created entry."""
    path = path.absolute()
    with ExitStack() as handles:
        parent = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        handles.callback(os.close, parent)
        parts = path.parts[1:]
        for index, part in enumerate(parts):
            try:
                os.mkdir(part, dir_fd=parent)
            except FileExistsError:
                if index == len(parts) - 1:
                    raise
            else:
                os.fsync(parent)
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            handles.callback(os.close, parent)
        yield _AuditDirectory(path, parent)


def _require_identity(directory: _AuditDirectory) -> None:
    current = directory.path.lstat()
    expected = os.fstat(directory.descriptor)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ValidationError("Instrument audit directory identity changed")


def _write_record(directory: _AuditDirectory, name: str, payload: bytes) -> None:
    """Publish once and sync both record and directory before any dependent request."""
    _require_identity(directory)
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory.descriptor,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(directory.descriptor)
    _require_identity(directory)


def collect_instrument_metadata(
    service: "DataService", request: InstrumentMetadataRequest
) -> InstrumentMetadataObservation:
    """Audit before provider resolution, retaining possible exposure on every failure."""
    if (
        request.purpose != "discovery"
        or request.arm_id != PROBE_ARM
        or request.symbols != MATRIX_SYMBOLS
        or request.final_portfolio_review
    ):
        raise ValidationError("Unapproved instrument-metadata declaration")
    environment = service.settings.oanda_environment
    if environment not in ("practice", "live"):
        raise ValidationError("Unapproved OANDA environment")
    directory = service._research_log_root / "instrument_metadata" / request.attempt_id
    with _claim_directory(directory) as audit:
        return _collect(service, request, audit)


def _collect(
    service: "DataService", request: InstrumentMetadataRequest, directory: _AuditDirectory
) -> InstrumentMetadataObservation:
    host = (
        "api-fxpractice.oanda.com"
        if service.settings.oanda_environment == "practice"
        else "api-fxtrade.oanda.com"
    )
    receipt = request.model_dump(mode="json") | {
        "requested_at": datetime.now(UTC).isoformat(),
        "provider": "oanda",
        "kind": "instrument_metadata",
        "maximum_gets": 1,
        "potential_exposure": True,
        "audit_directory": str(directory.path),
        "endpoint": f"https://{host}/v3/accounts/<redacted>/instruments",
        "query": {"instruments": ",".join(MATRIX_SYMBOLS)},
    }
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
    audit_bytes = json.dumps(
        {
            "kind": "instrument_metadata",
            "attempt_id": request.attempt_id,
            "requested_at": receipt["requested_at"],
            "purpose": request.purpose,
            "arm_id": request.arm_id,
            "symbols": request.symbols,
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
        sort_keys=True,
    ).encode()
    _write_record(directory, "receipt.json", receipt_bytes)
    _write_record(directory, "request.json", audit_bytes)
    expected = {"receipt.json": receipt_bytes, "request.json": audit_bytes}
    _reconcile(directory, expected)
    source = service._get_provider("oanda")
    if not isinstance(source, InstrumentMetadataProvider):
        raise ProviderError("Provider does not support instrument metadata probes")
    observation = source.probe_instrument_metadata()
    _reconcile(directory, expected)
    outcome = {
        "attempt_id": request.attempt_id,
        "observation": observation.model_dump(mode="json", by_alias=True, exclude_none=True),
        "observation_time_basis": "local_receipt_not_provider_as_of",
    }
    if observation.observed_at is None:
        outcome["failed_at"] = datetime.now(UTC).isoformat()
    _write_record(directory, "outcome.json", json.dumps(outcome, sort_keys=True).encode())
    return observation


def _reconcile(directory: _AuditDirectory, expected: dict[str, bytes]) -> None:
    """Require exact pre-request records; never accept altered or extra audit events."""
    _require_identity(directory)
    if set(os.listdir(directory.descriptor)) != set(expected):
        raise ValidationError("Instrument metadata audit records disagree")
    for name, payload in expected.items():
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory.descriptor)
        except OSError:
            raise ValidationError("Instrument metadata audit records disagree") from None
        with os.fdopen(descriptor, "rb") as record:
            if record.read() != payload:
                raise ValidationError("Instrument metadata audit records disagree")
    _require_identity(directory)

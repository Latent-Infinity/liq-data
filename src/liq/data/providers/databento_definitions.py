"""Default-deny, one-shot opaque definition capture; no research source admission."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx

from liq.data.lockbox import FoldWindows, LockboxGuard, LockboxLedger
from liq.data.providers._databento_transport import (
    BoundedResponse,
    MetadataRefused,
    fetch_bounded,
)

DEFINITION_DATASET: Final = "databento_xnas_instrument_definitions"
DEFINITION_ARM: Final = "classical_ranker_definition_source_char"
DEFINITION_LEDGER: Final = LockboxLedger(
    version="classical-ranker-definition-characterization-2026-10-03",
    datasets=MappingProxyType(
        {
            DEFINITION_DATASET: FoldWindows(
                characterization=(date(2018, 5, 1), date(2018, 5, 1)),
                lockbox_start=date(2026, 1, 1),
                allowed_purposes=frozenset({"characterization"}),
            ),
        }
    ),
)
_ENDPOINT: Final = "https://hist.databento.com/v0/timeseries.get_range"
_MAX_RESPONSE_BYTES: Final = 4 * 1024 * 1024
_REQUEST_SECONDS = 20.0


class DefinitionRefused(MetadataRefused):
    """Safe refusal; no raw records, key or provider error text."""


@dataclass(frozen=True, slots=True)
class DefinitionRequest:
    dataset: str = "XNAS.ITCH"
    start: str = "2018-05-01T00:00:00Z"
    end: str = "2018-05-02T00:00:00Z"
    limit: int = 1000

    def parameters(self) -> dict[str, str]:
        if (
            self.dataset != "XNAS.ITCH"
            or self.start != "2018-05-01T00:00:00Z"
            or self.end != "2018-05-02T00:00:00Z"
            or type(self.limit) is not int
            or self.limit != 1000
        ):
            raise DefinitionRefused("request_scope")
        return {
            "dataset": self.dataset,
            "schema": "definition",
            "symbols": "ALL_SYMBOLS",
            "stype_in": "raw_symbol",
            "stype_out": "instrument_id",
            "start": self.start,
            "end": self.end,
            "limit": str(self.limit),
            "encoding": "json",
            "compression": "none",
            "pretty_px": "false",
            "pretty_ts": "false",
            "map_symbols": "false",
        }


@dataclass(frozen=True, slots=True)
class AuthorityPin:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class DefinitionAuthority:
    """Trusted documentary identities, populated only after independent review."""

    execution_id: str
    capture_root: Path
    human: AuthorityPin
    oracle: AuthorityPin
    quote: AuthorityPin
    source_pins: tuple[AuthorityPin, ...]
    api_key_sha256: str
    estimated_usd: Decimal
    estimate_ceiling_usd: Decimal
    quote_valid_until: datetime
    uncapped_billing_acknowledged: bool


def authority_digest(authority: DefinitionAuthority) -> str:
    """Bind a parent-held authorization to the child's exact frozen registry entry."""
    body = json.dumps(asdict(authority), sort_keys=True, default=str).encode()
    return hashlib.sha256(body).hexdigest()


@dataclass(frozen=True, slots=True)
class DefinitionCapture:
    response: BoundedResponse = field(repr=False)
    receipt: bytes = field(repr=False)


# Neither caller dictionaries nor cheap quotes activate this route.
_AUTHORITIES: Mapping[str, DefinitionAuthority] = MappingProxyType({})


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    # A legitimate read can update access time; content/ownership changes still refuse.
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _read_pin(pin: AuthorityPin) -> bytes:
    """Hold bounded regular bytes through no-follow directory descriptors."""
    if not pin.path.is_absolute():
        raise DefinitionRefused("authority_pin")
    directory = os.open(pin.path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in pin.path.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            pin.path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or not 0 < before.st_size <= _MAX_RESPONSE_BYTES
            ):
                raise DefinitionRefused("authority_pin")
            body = source.read(_MAX_RESPONSE_BYTES + 1)
            after = os.fstat(source.fileno())
            if (
                not _same_file_state(before, after)
                or len(body) != before.st_size
                or hashlib.sha256(body).hexdigest() != pin.sha256
            ):
                raise DefinitionRefused("authority_pin")
            return body
    except OSError as exc:
        raise DefinitionRefused("authority_pin") from exc
    finally:
        os.close(directory)


def _validate_authority(authority: DefinitionAuthority) -> None:
    now = datetime.now(UTC)
    if (
        authority.quote_valid_until.tzinfo is None
        or authority.quote_valid_until.utcoffset() != UTC.utcoffset(now)
        or now >= authority.quote_valid_until
    ):
        raise DefinitionRefused("stale_quote")
    if (
        not authority.estimated_usd.is_finite()
        or not authority.estimate_ceiling_usd.is_finite()
        or authority.estimated_usd < 0
        or not 0 < authority.estimate_ceiling_usd <= Decimal("1")
        or authority.estimated_usd > authority.estimate_ceiling_usd
        or authority.uncapped_billing_acknowledged is not True
    ):
        raise DefinitionRefused("billing_authority")
    if not authority.source_pins:
        raise DefinitionRefused("missing_source_pins")
    for pin in (authority.human, authority.oracle, authority.quote, *authority.source_pins):
        _read_pin(pin)


def _root_identity(root: Path) -> tuple[int, int]:
    if not root.is_absolute() or root.resolve() != root:
        raise DefinitionRefused("capture_root")
    try:
        info = root.lstat()
    except OSError as exc:
        raise DefinitionRefused("capture_root") from exc
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise DefinitionRefused("capture_root")
    return info.st_dev, info.st_ino


def _write_private(root: Path, name: str, body: bytes) -> tuple[int, int]:
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
            identity = os.fstat(output.fileno())
        os.fsync(directory)
        return identity.st_dev, identity.st_ino
    finally:
        os.close(directory)


def _load_api_key() -> str:
    from liq.data.settings import LiqDataSettings

    return LiqDataSettings().databento_api_key or ""


def fetch_definitions(
    request: DefinitionRequest,
    *,
    authority_id: str,
    guard: LockboxGuard,
    purpose: str,
    arm_id: str,
    final_portfolio_review: bool = False,
    expected_authority_sha256: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DefinitionCapture:
    """Retain a single authorized opaque capture, never validated market records.

    The estimate ceiling is a local precondition, not an enforceable invoice cap.
    The fixed private root consumes the authority once, including failed attempts.
    """
    if type(request) is not DefinitionRequest:
        raise DefinitionRefused("request_scope")
    parameters = request.parameters()
    authority = _AUTHORITIES.get(authority_id)
    if authority is None or authority.execution_id != authority_id:
        raise DefinitionRefused("unknown_authority")
    held_authority_sha256 = authority_digest(authority)
    if expected_authority_sha256 is not None and expected_authority_sha256 != held_authority_sha256:
        raise DefinitionRefused("parent_authority")
    if (
        purpose != "characterization"
        or arm_id != DEFINITION_ARM
        or final_portfolio_review
        or type(guard) is not LockboxGuard
        or guard.ledger is not DEFINITION_LEDGER
        or guard.usage_log_path != authority.capture_root / "lockbox_usage.jsonl"
    ):
        raise DefinitionRefused("research_scope")
    _validate_authority(authority)
    root = authority.capture_root
    root_identity = _root_identity(root)
    if any(root.iterdir()):
        raise DefinitionRefused("consumed")
    try:
        _write_private(
            root,
            "consumed.json",
            json.dumps(
                {
                    "execution_id": authority_id,
                    "parameters": parameters,
                    "created_at": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
            ).encode(),
        )
    except FileExistsError as exc:
        raise DefinitionRefused("consumed") from exc
    _write_private(root, "lockbox_usage.jsonl", b"")
    guard.assert_period_allowed(
        DEFINITION_DATASET,
        date(2018, 5, 1),
        date(2018, 5, 1),
        purpose=purpose,
        arm_id=arm_id,
        final_portfolio_review=False,
    )
    _validate_authority(authority)
    api_key = _load_api_key()
    if not api_key or hashlib.sha256(api_key.encode()).hexdigest() != authority.api_key_sha256:
        raise DefinitionRefused("credential_context")

    def witness() -> None:
        if (
            _AUTHORITIES.get(authority_id) is not authority
            or authority_digest(authority) != held_authority_sha256
        ):
            raise DefinitionRefused("parent_authority")
        _validate_authority(authority)
        if _root_identity(root) != root_identity:
            raise DefinitionRefused("capture_root")
        _write_private(
            root,
            "attempt.json",
            json.dumps(
                {
                    "execution_id": authority_id,
                    "ordinal": 1,
                    "method": "POST",
                    "endpoint": _ENDPOINT,
                    "parameters": parameters,
                    "entered_at": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
            ).encode(),
        )

    try:
        response = fetch_bounded(
            method="POST",
            endpoint=_ENDPOINT,
            parameters=parameters,
            api_key=api_key,
            on_http_attempt=witness,
            max_response_bytes=_MAX_RESPONSE_BYTES,
            deadline_seconds=_REQUEST_SECONDS,
            transport=transport,
        )
    except MetadataRefused as exc:
        raise DefinitionRefused(
            exc.reason, http_status=exc.http_status, received_bytes=exc.received_bytes
        ) from exc
    # Even refusal responses survive; no interpretation precedes physical retention.
    if _root_identity(root) != root_identity:
        raise DefinitionRefused("capture_root")
    _write_private(root, "response.raw", response.body)
    nonempty_lines = sum(bool(line.strip()) for line in response.body.split(b"\n"))
    receipt = json.dumps(
        {
            "execution_id": authority_id,
            "parameters": parameters,
            "http_status": response.http_status,
            "content_type": response.content_type,
            "content_encoding": response.content_encoding,
            "bytes": len(response.body),
            "body_sha256": hashlib.sha256(response.body).hexdigest(),
            "nonempty_lines": nonempty_lines,
            "schema_interpretation": "UNVERIFIED",
            "completed_at": datetime.now(UTC).isoformat(),
        },
        sort_keys=True,
    ).encode()
    _write_private(root, "response.json", receipt)
    _validate_authority(authority)
    if response.http_status != 200:
        raise DefinitionRefused(
            "http_refused", http_status=response.http_status, received_bytes=len(response.body)
        )
    if nonempty_lines > request.limit:
        raise DefinitionRefused(
            "record_limit", http_status=response.http_status, received_bytes=len(response.body)
        )
    return DefinitionCapture(response, receipt)

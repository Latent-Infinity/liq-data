"""Fixed engineering-read declarations, separate from consumed acquisition."""

import fcntl
import os
import re
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from liq.data.alpaca_momentum_numerical_files import (
    Original,
    digest,
    hold_file,
    private_root,
    read_pin,
    require_pin,
    write_once,
)
from liq.data.alpaca_momentum_numerical_models import (
    Invocation,
    MomentumNumericalRead,
    MomentumNumericalRefused,
    NumericalAuthority,
    NumericalContract,
    NumericalRuntime,
    ReadReservation,
    SourceReceipt,
)
from liq.data.alpaca_momentum_numerical_runtime import check_runtime
from liq.data.lockbox import CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1, LockboxGuard


def check_numerical_guard(request: MomentumNumericalRead, guard: LockboxGuard) -> None:
    """Check the finite declaration before any source or guard-log mutation.

    General dev-smoke reads are fold-exempt. This boundary therefore enforces
    the exact period itself; the ledger alone cannot establish that constraint.
    Authority adoption and the ordered budget reservation must still precede
    source loading. This function neither reads a source nor claims a read.
    """
    try:
        MomentumNumericalRead.model_validate(request)
    except ValidationError as exc:
        raise MomentumNumericalRefused("numerical_request") from exc
    if guard.ledger != CLASSICAL_RANKER_MOMENTUM_NUMERICAL_LEDGER_V1:
        raise MomentumNumericalRefused("numerical_ledger")


@dataclass(frozen=True, slots=True)
class NumericalBinding:
    contract: NumericalContract
    runtime: NumericalRuntime
    authority: NumericalAuthority
    invocation: Invocation
    contract_sha256: str
    authority_sha256: str
    originals: tuple[Original, ...]
    source: SourceReceipt
    usage_identity: tuple[int, int]
    fixture_identity: tuple[int, int]

    def check(self) -> None:
        for original in self.originals:
            original.check()
        if (
            private_root(Path(self.contract.usage_root)) != self.usage_identity
            or private_root(Path(self.contract.fixture_root)) != self.fixture_identity
        ):
            raise MomentumNumericalRefused("scope_root_changed")
        for pin in self.contract.source_lineage:
            read_pin(pin)
        check_runtime(self.runtime, self.contract, self.contract_sha256, self.invocation)


@contextmanager
def adopt_numerical_scope(
    contract_path: Path, authority_path: Path, expected: tuple[str, str], invocation: Invocation
) -> Iterator[NumericalBinding]:
    """Adopt independently held hashes; no counter or source body is opened."""
    with ExitStack() as stack:
        cp = stack.enter_context(hold_file(contract_path, 1048576))
        ap = stack.enter_context(hold_file(authority_path, 65536))
        if (digest(cp.raw), digest(ap.raw)) != expected:
            raise MomentumNumericalRefused("parent_hash_binding")
        try:
            contract = NumericalContract.model_validate_json(cp.raw)
            authority = NumericalAuthority.model_validate_json(ap.raw)
            if authority.contract_sha256 != expected[0]:
                raise MomentumNumericalRefused("authority_contract")
            oracle = stack.enter_context(
                hold_file(Path(authority.oracle_decision.path), 65536, private=False)
            )
            require_pin(oracle, authority.oracle_decision)
            text = oracle.raw.decode("utf-8")
            if expected[0] not in text or not re.search(
                r"^(?:Recommendation|Decision): APPROVE\b", text, re.MULTILINE
            ):
                raise MomentumNumericalRefused("oracle_decision")
            environment = stack.enter_context(hold_file(Path(contract.runtime.path), 4194304))
            require_pin(environment, contract.runtime)
            runtime = NumericalRuntime.model_validate_json(environment.raw)
            source = stack.enter_context(
                hold_file(Path(contract.source_receipt.path), 131072, private=False)
            )
            require_pin(source, contract.source_receipt)
            if contract.source_receipt.sha256 != (
                "75d9a768278e03cd1a853e01a3ea88af5ecd51257f0155ba5b30bd298c4851c6"
            ):
                raise MomentumNumericalRefused("source_receipt_admission")
            decoded = SourceReceipt.model_validate_json(source.raw)
            binding = NumericalBinding(
                contract,
                runtime,
                authority,
                invocation,
                *expected,
                (cp, ap, oracle, environment, source),
                decoded,
                private_root(Path(contract.usage_root)),
                private_root(Path(contract.fixture_root)),
            )
            binding.check()
            yield binding
            binding.check()
        except (ValidationError, UnicodeError) as exc:
            raise MomentumNumericalRefused("scope_metadata") from exc


def reserve_numerical_read(binding: NumericalBinding) -> ReadReservation:
    """Reserve once in the frozen sequence before any retained page is opened."""
    binding.check()
    root = Path(binding.contract.usage_root)
    path = root / "numerical-usage.jsonl"
    try:
        descriptor = os.open(
            path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
    except OSError as exc:
        raise MomentumNumericalRefused("usage_open") from exc
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size > 65536
    ):
        os.close(descriptor)
        raise MomentumNumericalRefused("usage_policy")
    with os.fdopen(descriptor, "r+", encoding="utf-8") as usage:
        fcntl.flock(usage, fcntl.LOCK_EX)
        raw = usage.read(65537)
        if len(raw.encode()) > 65536:
            raise MomentumNumericalRefused("usage_byte_budget")
        try:
            rows = tuple(ReadReservation.model_validate_json(line) for line in raw.splitlines())
        except ValidationError as exc:
            raise MomentumNumericalRefused("usage_metadata") from exc
        expected_order = binding.contract.invocations
        if (
            len(rows) >= len(expected_order)
            or tuple(row.invocation for row in rows) != expected_order[: len(rows)]
            or expected_order[len(rows)] != binding.invocation
            or any(
                (row.contract_sha256, row.authority_sha256)
                != (binding.contract_sha256, binding.authority_sha256)
                for row in rows
            )
        ):
            raise MomentumNumericalRefused("usage_order_or_budget")
        if rows:
            previous = root / f"{rows[-1].invocation}-complete.json"
            with hold_file(previous, 65536) as original:
                completion = ReadReservation.model_validate_json(original.raw)
                if completion != rows[-1]:
                    raise MomentumNumericalRefused("previous_completion")
        row = ReadReservation(
            contract_sha256=binding.contract_sha256,
            authority_sha256=binding.authority_sha256,
            invocation=binding.invocation,
        )
        write_once(root, f"{binding.invocation}-reserved.json", row.model_dump_json().encode())
        usage.write(row.model_dump_json() + "\n")
        usage.flush()
        os.fsync(usage.fileno())
        binding.check()
        return row


def complete_numerical_read(binding: NumericalBinding, row: ReadReservation) -> tuple[int, ...]:
    """Record a worker's successful reconciliation; do not retry a failed reservation."""
    binding.check()
    if row != ReadReservation(
        contract_sha256=binding.contract_sha256,
        authority_sha256=binding.authority_sha256,
        invocation=binding.invocation,
    ):
        raise MomentumNumericalRefused("completion_binding")
    with hold_file(
        Path(binding.contract.usage_root) / f"{binding.invocation}-decoded.json", 65536
    ) as original:
        if ReadReservation.model_validate_json(original.raw) != row:
            raise MomentumNumericalRefused("decoded_binding")
    return write_once(
        Path(binding.contract.usage_root),
        f"{binding.invocation}-complete.json",
        row.model_dump_json().encode(),
    )

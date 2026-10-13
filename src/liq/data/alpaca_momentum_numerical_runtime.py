"""Source and environment reconciliation for the retained momentum case."""

import importlib
import importlib.metadata
import sys
import zoneinfo
from pathlib import Path
from typing import Literal

from liq.data.alpaca_momentum_numerical_files import digest, read_pin
from liq.data.alpaca_momentum_numerical_models import (
    Digest,
    FilePin,
    Invocation,
    MetadataModel,
    MomentumNumericalRefused,
    NumericalContract,
    NumericalRuntime,
)


class ReferenceBinding(MetadataModel):
    contract_sha256: str
    value: str


class ReferencePublication(MetadataModel):
    invocation: Literal["reference"]
    contract_sha256: Digest
    authority_sha256: Digest
    reference_sha256: Digest
    reference_identity: tuple[int, ...]


def check_test_amendment(
    contract: NumericalContract, contract_sha256: str, invocation: Invocation
) -> None:
    before = read_pin(contract.original_test)
    token = f'REFERENCE_SHA256 = "{contract.reference_placeholder}"'.encode()
    if before.count(token) != 1:
        raise MomentumNumericalRefused("reference_template")
    expected = before
    if invocation != "reference":
        path = Path(contract.reference_path)
        from liq.data.alpaca_momentum_numerical_files import hold_file

        with hold_file(path, 65536, private=False) as original:
            reference = ReferenceBinding.model_validate_json(original.raw)
            if reference.contract_sha256 != contract_sha256:
                raise MomentumNumericalRefused("reference_contract")
            reference_digest = digest(original.raw)
            reference_identity = original.identity
        with hold_file(
            Path(contract.usage_root) / "reference" / "accepted.json", 65536
        ) as original:
            publication = ReferencePublication.model_validate_json(original.raw)
            if (
                publication.contract_sha256 != contract_sha256
                or publication.reference_sha256 != reference_digest
                or publication.reference_identity != reference_identity
            ):
                raise MomentumNumericalRefused("reference_publication")
        replacement = f'REFERENCE_SHA256 = "{reference_digest}"'.encode()
        expected = before.replace(token, replacement, 1)
    path = Path(contract.current_test)
    from liq.data.alpaca_momentum_numerical_files import hold_file

    with hold_file(path, 1048576, private=False) as original:
        if original.raw != expected:
            raise MomentumNumericalRefused("test_amendment")


def check_runtime(
    runtime: NumericalRuntime,
    contract: NumericalContract,
    contract_sha256: str,
    invocation: Invocation,
) -> None:
    if Path.cwd() != Path(runtime.cwd):
        raise MomentumNumericalRefused("runtime_cwd")
    invocation_path = Path(runtime.interpreter_invocation)
    if invocation_path != Path(sys.executable).absolute() or invocation_path.resolve() != Path(
        runtime.interpreter.path
    ):
        raise MomentumNumericalRefused("runtime_interpreter")
    if tuple(zoneinfo.TZPATH) != runtime.timezone_search_path:
        raise MomentumNumericalRefused("runtime_timezone_path")
    check_test_amendment(contract, contract_sha256, invocation)
    for pin in (*runtime.code_pins, runtime.interpreter, runtime.timezone):
        if pin.path != contract.current_test:
            read_pin(pin)
    for name in runtime.bootstrap_modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise MomentumNumericalRefused("runtime_bootstrap") from exc
    for origin in runtime.origins:
        try:
            module = importlib.import_module(origin.module)
        except ImportError as exc:
            raise MomentumNumericalRefused("runtime_import") from exc
        if not module.__file__ or Path(module.__file__).resolve() != Path(origin.path):
            raise MomentumNumericalRefused("runtime_origin")
        if origin.path != contract.current_test:
            read_pin(FilePin.model_validate(origin.model_dump(exclude={"module"})))
    for name, version in runtime.package_versions:
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise MomentumNumericalRefused("runtime_package") from exc
        if actual != version:
            raise MomentumNumericalRefused("runtime_package")

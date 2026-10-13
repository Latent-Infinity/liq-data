"""Nonfinancial import/publication controls for the frozen runtime."""

import importlib.metadata
from pathlib import Path

import pytest
from momentum_numerical_test_support import amendment, metadata_scope

from liq.data.alpaca_momentum_numerical_files import pin_file
from liq.data.alpaca_momentum_numerical_models import (
    ImportPin,
    MomentumNumericalRefused,
    NumericalContract,
    NumericalRuntime,
)
from liq.data.alpaca_momentum_numerical_runtime import check_runtime


def test_missing_distribution_is_typed_refusal(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    contract = NumericalContract.model_validate_json(case.contract.read_bytes())
    runtime = NumericalRuntime.model_validate_json(Path(contract.runtime.path).read_bytes())
    runtime = runtime.model_copy(update={"package_versions": (("missing-test-package", "TEST"),)})

    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError("missing-test-package")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(MomentumNumericalRefused, match="runtime_package"):
        check_runtime(runtime, contract, case.expected()[0], "reference")


def test_import_origin_mismatch_is_refused(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    contract = NumericalContract.model_validate_json(case.contract.read_bytes())
    runtime = NumericalRuntime.model_validate_json(Path(contract.runtime.path).read_bytes())
    origin = ImportPin(module="json", **pin_file(case.code).model_dump())
    runtime = runtime.model_copy(update={"origins": (origin,)})
    with pytest.raises(MomentumNumericalRefused, match="runtime_origin"):
        check_runtime(runtime, contract, case.expected()[0], "reference")


def test_missing_bootstrap_import_is_typed_refusal(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    contract = NumericalContract.model_validate_json(case.contract.read_bytes())
    runtime = NumericalRuntime.model_validate_json(Path(contract.runtime.path).read_bytes())
    runtime = runtime.model_copy(update={"bootstrap_modules": ("missing_numerical_test_module",)})
    with pytest.raises(MomentumNumericalRefused, match="runtime_bootstrap"):
        check_runtime(runtime, contract, case.expected()[0], "reference")


def test_disk_reference_and_literal_cannot_replace_generator_witness(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    amendment(case)
    before = pin_file(case.reference).sha256
    case.reference.write_text(
        case.reference.read_text().replace("CONTROL_ONLY", "ALTERED_CONTROL_ONLY")
    )
    case.current_test.write_text(
        case.current_test.read_text().replace(before, pin_file(case.reference).sha256)
    )
    contract = NumericalContract.model_validate_json(case.contract.read_bytes())
    runtime = NumericalRuntime.model_validate_json(Path(contract.runtime.path).read_bytes())
    with pytest.raises(MomentumNumericalRefused, match="reference_publication"):
        check_runtime(runtime, contract, case.expected()[0], "focused")

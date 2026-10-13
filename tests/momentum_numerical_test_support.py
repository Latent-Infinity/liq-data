"""Nonexecuting metadata controls; the temporary source directory is empty."""

import json
import sys
import zoneinfo
from dataclasses import dataclass
from pathlib import Path

from liq.data.alpaca_momentum_numerical_files import pin_file, state
from liq.data.alpaca_momentum_numerical_models import (
    NumericalAuthority,
    NumericalContract,
    NumericalRuntime,
)


def write(path: Path, raw: bytes) -> Path:
    path.write_bytes(raw)
    path.chmod(0o600)
    return path


@dataclass(frozen=True, slots=True)
class MetadataScope:
    contract: Path
    authority: Path
    oracle: Path
    code: Path
    source_root: Path
    usage_root: Path
    reference: Path
    current_test: Path

    def expected(self) -> tuple[str, str]:
        return pin_file(self.contract).sha256, pin_file(self.authority).sha256


def metadata_scope(tmp_path: Path, monkeypatch) -> MetadataScope:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "EMPTY_NONFINANCIAL_SOURCE"
    usage = tmp_path / "NONEXECUTING_TEST_USAGE"
    source.mkdir(mode=0o700)
    usage.mkdir(mode=0o700)
    code = write(tmp_path / "metadata-code.py", b"# source-free metadata control\n")
    before = write(
        tmp_path / "before-test.py",
        ('REFERENCE_SHA256 = "' + "0" * 64 + '"\n').encode(),
    )
    current = write(tmp_path / "current-test.py", before.read_bytes())
    capture = (
        Path(__file__).resolve().parents[2]
        / "liq-experiments/artifacts/intraday_slate/classical_cross_sectional_ranker"
        / "momentum-source-capture-2026-10-03/receipt.json"
    )
    # This public receipt contains timestamps/accounting, not source prices.
    receipt = write(tmp_path / "public-source-receipt.json", capture.read_bytes())
    runtime = NumericalRuntime(
        cwd=str(tmp_path),
        interpreter=pin_file(Path(sys.executable).resolve()),
        interpreter_invocation=str(Path(sys.executable).absolute()),
        timezone=pin_file(code),
        timezone_search_path=tuple(zoneinfo.TZPATH),
        code_pins=(pin_file(code),),
        origins=(),
        bootstrap_modules=(),
        package_versions=(),
    )
    env = write(tmp_path / "metadata-runtime.json", runtime.model_dump_json().encode())
    reference = tmp_path / "NONFINANCIAL_REFERENCE_METADATA.json"
    contract = NumericalContract(
        fixture_root=str(source),
        usage_root=str(usage),
        source_receipt=pin_file(receipt),
        source_lineage=(pin_file(code),),
        runtime=pin_file(env),
        reference_path=str(reference),
        original_test=pin_file(before),
        current_test=str(current),
    )
    cp = write(tmp_path / "TEST-contract.json", contract.model_dump_json().encode())
    oracle = write(
        tmp_path / "NONEXECUTING_TEST_ORACLE.md",
        f"Recommendation: APPROVE\nNonexecuting metadata only.\n{pin_file(cp).sha256}\n".encode(),
    )
    authority = NumericalAuthority(
        decision="APPROVE", contract_sha256=pin_file(cp).sha256, oracle_decision=pin_file(oracle)
    )
    ap = write(tmp_path / "TEST-authority.json", authority.model_dump_json().encode())
    return MetadataScope(cp, ap, oracle, code, source, usage, reference, current)


def amendment(case: MetadataScope) -> None:
    write(
        case.reference,
        json.dumps(
            {
                "contract_sha256": pin_file(case.contract).sha256,
                "value": "CONTROL_ONLY_NONFINANCIAL_METADATA",
            }
        ).encode(),
    )
    token = "0" * 64
    case.current_test.write_text(
        case.current_test.read_text().replace(token, pin_file(case.reference).sha256)
    )
    root = case.usage_root / "reference"
    root.mkdir(mode=0o700)
    write(
        root / "accepted.json",
        json.dumps(
            {
                "invocation": "reference",
                "contract_sha256": pin_file(case.contract).sha256,
                "authority_sha256": pin_file(case.authority).sha256,
                "reference_sha256": pin_file(case.reference).sha256,
                "reference_identity": state(case.reference.stat()),
            }
        ).encode(),
    )

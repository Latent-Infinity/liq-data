"""Nonexecuting authorization/budget controls with an empty source directory."""

import json
import os

import pytest
from momentum_numerical_test_support import amendment, metadata_scope, write

from liq.data.alpaca_momentum_numerical_files import digest
from liq.data.alpaca_momentum_numerical_models import MomentumNumericalRefused
from liq.data.alpaca_momentum_numerical_scope import (
    adopt_numerical_scope,
    complete_numerical_read,
    reserve_numerical_read,
)


def test_reservation_fifo_is_typed_refusal_before_wrapping(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    os.mkfifo(case.usage_root / "numerical-usage.jsonl", 0o600)
    with (
        adopt_numerical_scope(
            case.contract, case.authority, case.expected(), "reference"
        ) as binding,
        pytest.raises(MomentumNumericalRefused, match="usage_policy"),
    ):
        reserve_numerical_read(binding)
    assert not (case.usage_root / "reference-reserved.json").exists()


def test_missing_authority_or_parent_mismatch_opens_no_source(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    expected = case.expected()
    with (
        pytest.raises(MomentumNumericalRefused, match="parent_hash_binding"),
        adopt_numerical_scope(case.contract, case.authority, ("0" * 64, expected[1]), "reference"),
    ):
        pytest.fail("wrong parent hash adopted")
    case.authority.unlink()
    with (
        pytest.raises(MomentumNumericalRefused, match="file_open"),
        adopt_numerical_scope(case.contract, case.authority, expected, "reference"),
    ):
        pytest.fail("missing authority adopted")
    assert not tuple(case.source_root.iterdir())


@pytest.mark.parametrize("target", ["contract", "authority", "oracle", "code"])
def test_documentary_mutation_refuses_before_or_during_teardown(
    tmp_path, monkeypatch, target: str
) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    with (
        pytest.raises(MomentumNumericalRefused),
        adopt_numerical_scope(
            case.contract, case.authority, case.expected(), "reference"
        ) as binding,
    ):
        path = getattr(case, target)
        path.write_bytes(path.read_bytes() + b" ")
        binding.check()
    assert not tuple(case.source_root.iterdir())


def test_case_reservation_is_once_and_consumed_without_reopening(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    with adopt_numerical_scope(
        case.contract, case.authority, case.expected(), "reference"
    ) as binding:
        row = reserve_numerical_read(binding)
        assert row.planned_decoded_daily_records == 251
        with pytest.raises(MomentumNumericalRefused, match="usage_order_or_budget"):
            reserve_numerical_read(binding)
    assert not tuple(case.source_root.iterdir())


def test_completion_cannot_advance_without_a_decoded_case(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    with adopt_numerical_scope(
        case.contract, case.authority, case.expected(), "reference"
    ) as binding:
        row = reserve_numerical_read(binding)
        with pytest.raises(MomentumNumericalRefused, match="file_open"):
            complete_numerical_read(binding, row)


def test_attempt_order_requires_previous_completion(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    with adopt_numerical_scope(
        case.contract, case.authority, case.expected(), "reference"
    ) as binding:
        reserve_numerical_read(binding)
    amendment(case)
    with (
        adopt_numerical_scope(case.contract, case.authority, case.expected(), "focused") as binding,
        pytest.raises(MomentumNumericalRefused, match="file_open"),
    ):
        reserve_numerical_read(binding)
    assert not tuple(case.source_root.iterdir())


def test_three_ordered_metadata_reservations_and_no_fourth(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    for invocation in ("reference", "focused", "full-feature"):
        with adopt_numerical_scope(
            case.contract, case.authority, case.expected(), invocation
        ) as binding:
            row = reserve_numerical_read(binding)
            # Protocol metadata only; this empty temporary case opens no page.
            write(case.usage_root / f"{invocation}-decoded.json", row.model_dump_json().encode())
            complete_numerical_read(binding, row)
        if invocation == "reference":
            amendment(case)
    with (
        adopt_numerical_scope(
            case.contract, case.authority, case.expected(), "full-feature"
        ) as binding,
        pytest.raises(MomentumNumericalRefused, match="usage_order_or_budget"),
    ):
        reserve_numerical_read(binding)
    rows = [
        json.loads(line)
        for line in (case.usage_root / "numerical-usage.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert sum(row["planned_decoded_daily_records"] for row in rows) == 753
    assert not tuple(case.source_root.iterdir())


def test_literal_reference_amendment_rejects_extra_edits(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    amendment(case)
    case.current_test.write_bytes(case.current_test.read_bytes() + b"# extra edit\n")
    with (
        pytest.raises(MomentumNumericalRefused, match="test_amendment"),
        adopt_numerical_scope(case.contract, case.authority, case.expected(), "focused"),
    ):
        pytest.fail("extra edit adopted")


def test_rejected_oracle_document_cannot_supply_authority(tmp_path, monkeypatch) -> None:
    case = metadata_scope(tmp_path, monkeypatch)
    rejected = f"Recommendation: REJECT\n{case.expected()[0]}\n".encode()
    write(case.oracle, rejected)
    raw = json.loads(case.authority.read_bytes())
    raw["oracle_decision"].update({"sha256": digest(rejected), "bytes": len(rejected)})
    write(case.authority, json.dumps(raw).encode())
    with (
        pytest.raises(MomentumNumericalRefused, match="oracle_decision"),
        adopt_numerical_scope(case.contract, case.authority, case.expected(), "reference"),
    ):
        pytest.fail("rejected decision adopted")

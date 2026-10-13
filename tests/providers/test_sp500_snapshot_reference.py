"""Reference identity/schema refusals without invented constituent evidence."""

import hashlib
from datetime import date

import pytest

from liq.data.providers import sp500_membership as module


def parse(raw: bytes, **overrides: object) -> list[str]:
    options = {
        "expected_git_blob": hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest(),
        "expected_bytes": len(raw),
        "observation_date": date(2023, 4, 21),
    }
    options.update(overrides)
    return module.parse_snapshot_reference(raw, **options)


@pytest.mark.parametrize("override", [{"expected_git_blob": "0" * 40}, {"expected_bytes": 0}])
def test_snapshot_identity_refuses_before_decoding(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="identity"):
        parse(b"\xff", **override)


def test_snapshot_missing_exact_date_refuses_without_carry_forward() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        parse(b"date,tickers\n")


def test_snapshot_duplicate_date_refuses_without_deduplication() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        parse(b'date,tickers\n2023-04-21,""\n2023-04-21,""\n')


def test_snapshot_malformed_member_field_refuses() -> None:
    with pytest.raises(ValueError, match="ticker"):
        parse(b'date,tickers\n2023-04-21,"../invalid"\n')


def test_snapshot_unrecognized_header_refuses() -> None:
    with pytest.raises(ValueError, match="header"):
        parse(b"unrecognized,columns\n")

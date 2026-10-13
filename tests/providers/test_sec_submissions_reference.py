"""Nonfinancial structural controls for a bounded submissions reference."""

from datetime import UTC, datetime

import pytest

from liq.data.providers.sec_edgar import parse_submissions_reference

FIELDS = ("form", "accessionNumber", "acceptanceDateTime", "filingDate", "items", "primaryDocument")


def empty_reference():
    return {
        "cik": "0001637459",
        "name": "Kraft Heinz Co",
        "filings": {"recent": {field: [] for field in FIELDS}, "files": []},
    }


def parse(payload):
    return parse_submissions_reference(
        payload,
        cik="0001637459",
        start=datetime(2023, 1, 1, tzinfo=UTC),
        end_exclusive=datetime(2023, 4, 25, tzinfo=UTC),
    )


def test_empty_recent_has_no_candidate_and_no_absence_claim():
    result = parse(empty_reference())
    assert result["selected_candidate"] is None
    assert result["recent_rows"] == 0
    assert result["historical_completeness_established"] is False


def test_reference_refuses_conflicting_issuer():
    payload = empty_reference()
    payload["cik"] = "0000000000"
    with pytest.raises(ValueError, match="issuer"):
        parse(payload)


@pytest.mark.parametrize("field", FIELDS)
def test_reference_requires_every_parallel_array(field):
    payload = empty_reference()
    del payload["filings"]["recent"][field]
    with pytest.raises(ValueError, match="parallel"):
        parse(payload)


def test_reference_refuses_mismatched_parallel_arrays():
    payload = empty_reference()
    payload["filings"]["recent"]["form"] = ["8-K"]
    with pytest.raises(ValueError, match="parallel"):
        parse(payload)


def test_reference_refuses_unbounded_archive_index():
    payload = empty_reference()
    payload["filings"]["files"] = [{}] * 1001
    with pytest.raises(ValueError, match="archive"):
        parse(payload)


def test_reference_refuses_foreign_archive_route():
    payload = empty_reference()
    payload["filings"]["files"] = [{"name": "../elsewhere.json"}]
    with pytest.raises(ValueError, match="archive"):
        parse(payload)

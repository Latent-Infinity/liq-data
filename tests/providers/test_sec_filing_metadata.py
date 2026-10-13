"""Opaque metadata grammar controls, never research or market observations."""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from liq.data.providers import sec_filing_metadata as metadata
from liq.data.providers.sec_edgar import SECEdgarProvider
from liq.data.providers.sec_filing_metadata import (
    MetadataRefusal,
    combine_metadata,
    parse_archive_metadata,
    parse_root_metadata,
    select_archive,
    select_ticker_lead,
)

CIK = "0000000001"
START = datetime(2018, 2, 1, tzinfo=UTC)
END = datetime(2018, 3, 1, tzinfo=UTC)
FORMS = frozenset({"8-K", "8-K/A", "10-K", "10-K/A"})


@pytest.mark.parametrize(
    "literal,tokens,spaces,empty",
    [
        ("5.03,8.01", ("5.03", "8.01"), 0, 0),
        (" 5.03 ,\t8.01\t", ("5.03", "8.01"), 4, 0),
        ("5.03,,8.01,", ("5.03", "8.01"), 0, 2),
        ("8.01,5.03,8.01", ("8.01", "5.03", "8.01"), 0, 0),
        ("", (), 0, 0),
        (" \t", (), 2, 1),
    ],
)
def test_lexical_adapter_preserves_native_text_order_duplicates_and_handling(
    literal, tokens, spaces, empty
):
    parsed = metadata.tokenize_native_items(literal)
    assert parsed.native_literal == literal
    assert parsed.tokens == tokens
    assert parsed.whitespace_count == spaces
    assert parsed.empty_segments == empty
    assert not parsed.unsupported_grammar
    payload = root()
    append_row(payload["filings"]["recent"], items=literal)
    reference = parse(payload).references[0]
    assert reference.native_items == literal
    assert reference.items == tokens
    assert reference.item_whitespace_count == spaces
    assert reference.item_empty_segments == empty


@pytest.mark.parametrize(
    "literal", ["2 .02", "2. 02", "٢.٠٢", "\u00a02.02", "bad", "1", "2.02;9.01"]
)
def test_lexical_adapter_never_repairs_unsupported_grammar(literal):
    parsed = metadata.tokenize_native_items(literal)
    assert parsed.native_literal == literal
    assert parsed.unsupported_grammar
    payload = root()
    append_row(payload["filings"]["recent"], items=literal, clock="2016-01-01T00:00:00Z")
    with pytest.raises(MetadataRefusal, match="invalid native item token"):
        parse(payload)


def test_lexical_adapter_classifies_interior_whitespace_without_removing_it():
    parsed = metadata.tokenize_native_items(" 2. 02 ")
    assert parsed.tokens == ("2. 02",)
    assert parsed.interior_whitespace
    assert parsed.unsupported_grammar
    assert parsed.whitespace_count == 2


def test_archive_descriptors_do_not_inspect_or_admit_recent_member():
    class NoSemanticRead:
        def __getitem__(self, _key):
            pytest.fail("recent member was semantically read")

        def get(self, _key, _default=None):
            pytest.fail("recent member was semantically read")

        def items(self):
            pytest.fail("recent member was semantically read")

    payload = root()
    payload["filings"]["recent"] = NoSemanticRead()
    payload["filings"]["files"] = [
        descriptor(2),
        descriptor(1, start="2016-01-01", end="2016-12-31"),
    ]
    result = metadata.parse_root_archive_descriptors(payload, cik=CIK)
    assert result.cik == CIK
    assert result.issuer_name == "OPAQUE METADATA CONTROL"
    assert [row.name for row in result.archives] == [descriptor(2)["name"], descriptor(1)["name"]]
    assert not hasattr(result, "references")
    assert (
        select_archive(result, start=START.date(), end_exclusive=END.date()).name
        == descriptor(2)["name"]
    )


def test_recent_anomaly_remains_refused_while_descriptors_validate():
    payload = root()
    append_row(payload["filings"]["recent"], items="OPAQUE_UNSUPPORTED")
    payload["filings"]["files"] = [descriptor()]
    with pytest.raises(MetadataRefusal, match="invalid native item token"):
        parse(payload)
    result = metadata.parse_root_archive_descriptors(payload, cik=CIK)
    assert len(result.archives) == 1
    assert not hasattr(result, "references")


@pytest.mark.parametrize(
    "fault",
    [
        "cik",
        "name",
        "filings",
        "files",
        "archive_name",
        "count_bool",
        "count_zero",
        "bounds",
        "duplicate",
        "overlap",
    ],
)
def test_independent_descriptor_refusals(fault):
    payload = root()
    payload["filings"]["files"] = [descriptor()]
    if fault == "cik":
        payload["cik"] = 2
    elif fault == "name":
        payload["name"] = None
    elif fault == "filings":
        payload["filings"] = []
    elif fault == "files":
        payload["filings"]["files"] = {}
    else:
        entry = payload["filings"]["files"][0]
        if fault == "archive_name":
            entry["name"] = "CIK0000000002-submissions-001.json"
        elif fault == "count_bool":
            entry["filingCount"] = True
        elif fault == "count_zero":
            entry["filingCount"] = 0
        elif fault == "bounds":
            entry["filingTo"] = "2017-01-01"
        elif fault == "duplicate":
            payload["filings"]["files"].append(entry.copy())
        elif fault == "overlap":
            payload["filings"]["files"].append(descriptor(2))
    with pytest.raises(MetadataRefusal):
        result = metadata.parse_root_archive_descriptors(payload, cik=CIK)
        select_archive(result, start=START.date(), end_exclusive=END.date())


def test_zero_archive_overlap_is_unresolved_locator_not_absence():
    payload = root()
    payload["filings"]["files"] = [descriptor(start="2016-01-01", end="2016-12-31")]
    result = metadata.parse_root_archive_descriptors(payload, cik=CIK)
    assert select_archive(result, start=START.date(), end_exclusive=END.date()) is None


FIELDS = ("form", "accessionNumber", "acceptanceDateTime", "filingDate", "items", "primaryDocument")


def arrays():
    return {field: [] for field in FIELDS}


def root():
    return {
        "cik": 1,
        "name": "OPAQUE METADATA CONTROL",
        "filings": {"recent": arrays(), "files": []},
    }


def append_row(data, *, sequence=1, form="8-K", clock="2018-02-01T00:00:00Z", items="5.03,8.01"):
    for key, value in zip(
        FIELDS,
        (form, f"0000000001-18-{sequence:06d}", clock, "2018-02-01", items, "opaque.htm"),
        strict=True,
    ):
        data[key].append(value)


def parse(payload):
    return parse_root_metadata(payload, cik=CIK, start=START, end_exclusive=END, forms=FORMS)


def descriptor(number=1, start="2018-01-01", end="2018-02-28"):
    return {
        "name": f"CIK{CIK}-submissions-{number:03d}.json",
        "filingCount": 1,
        "filingFrom": start,
        "filingTo": end,
    }


def locate(payload, maximum=16):
    return metadata.parse_root_filing_locators(
        payload, cik=CIK, start=START, end_exclusive=END, forms=FORMS, maximum=maximum
    )


def test_filing_locator_does_not_access_items_or_archive_descriptors():
    class LocatorColumns(dict):
        def get(self, key, default=None):
            assert key != "items", "item semantics accessed"
            return super().get(key, default)

    payload = root()
    append_row(payload["filings"]["recent"], items="unsupported opaque token")
    payload["filings"]["recent"] = LocatorColumns(payload["filings"]["recent"])
    payload["filings"]["files"] = object()
    batch = locate(payload)
    assert batch.issuer_name == "OPAQUE METADATA CONTROL"
    assert len(batch.references) == 1
    assert not hasattr(batch.references[0], "items")
    assert not hasattr(batch, "archives")
    payload["filings"]["recent"] = dict(payload["filings"]["recent"])
    with pytest.raises(MetadataRefusal, match="item token"):
        parse(payload)


def test_filing_locator_preserves_all_native_eligible_forms_and_order():
    payload = root()
    for sequence, form in enumerate(("10-K/A", "8-K", "8-K/A", "10-K"), 1):
        append_row(payload["filings"]["recent"], sequence=sequence, form=form)
    batch = locate(payload)
    assert [row.form for row in batch.references] == ["10-K/A", "8-K", "8-K/A", "10-K"]
    assert batch.references[0].acceptance_literal == "2018-02-01T00:00:00Z"
    assert batch.references[0].document_url.endswith("/000000000118000001/opaque.htm")


@pytest.mark.parametrize(
    "clock,count",
    [
        ("2018-01-31T23:59:59Z", 0),
        ("2018-02-01T00:00:00Z", 1),
        ("2018-03-01T00:00:00Z", 0),
        ("2018-02-01T01:00:00+01:00", 1),
    ],
)
def test_filing_locator_window(clock, count):
    payload = root()
    append_row(payload["filings"]["recent"], clock=clock)
    assert len(locate(payload).references) == count


@pytest.mark.parametrize("field", [name for name in FIELDS if name != "items"])
def test_filing_locator_requires_only_aligned_locator_columns(field):
    payload = root()
    append_row(payload["filings"]["recent"])
    payload["filings"]["recent"][field] = []
    with pytest.raises(MetadataRefusal, match="length mismatch"):
        locate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("acceptanceDateTime", "2018-02-01T00:00:00"),
        ("filingDate", "2018-02-30"),
        ("accessionNumber", "bad"),
        ("primaryDocument", "../unsafe.htm"),
    ],
)
def test_filing_locator_refuses_malformed_selected_identity(field, value):
    payload = root()
    append_row(payload["filings"]["recent"])
    payload["filings"]["recent"][field][0] = value
    with pytest.raises(MetadataRefusal):
        locate(payload)


def test_filing_locator_global_duplicate_refusal_and_no_truncation():
    payload = root()
    append_row(payload["filings"]["recent"])
    append_row(payload["filings"]["recent"], form="4")
    with pytest.raises(MetadataRefusal, match="duplicate accession"):
        locate(payload)
    payload["filings"]["recent"]["accessionNumber"][1] = "0000000001-18-000002"
    payload["filings"]["recent"]["primaryDocument"][1] = "renderer/opaque.xml"
    assert len(locate(payload).references) == 1
    append_row(payload["filings"]["recent"], sequence=3)
    with pytest.raises(MetadataRefusal, match="budget exceeded"):
        locate(payload, maximum=1)


def test_empty_filing_locator_and_root_mismatch():
    assert locate(root()).references == ()
    payload = root()
    payload["cik"] = 2
    with pytest.raises(MetadataRefusal, match="issuer mismatch"):
        locate(payload)


def test_empty_root_preserves_current_name_without_completeness_claim():
    result = parse(root())
    assert result.issuer_name == "OPAQUE METADATA CONTROL"
    assert result.references == ()
    assert result.accessions == ()
    assert result.archives == ()


def test_non_earnings_forms_and_amendments_preserve_native_order():
    payload = root()
    for number, form in enumerate(("8-K/A", "10-K", "8-K", "10-K/A"), 1):
        append_row(payload["filings"]["recent"], sequence=number, form=form)
    result = parse(payload)
    assert tuple(row.form for row in result.references) == ("8-K/A", "10-K", "8-K", "10-K/A")
    assert result.references[0].items == ("5.03", "8.01")
    assert result.references[0].acceptance_literal == "2018-02-01T00:00:00Z"
    assert result.references[0].acceptance_utc == START
    assert result.references[0].document_url.endswith("/opaque.htm")
    assert result.references[0].metadata_url == f"https://data.sec.gov/submissions/CIK{CIK}.json"


@pytest.mark.parametrize(
    "clock,count",
    [("2018-01-31T23:59:59Z", 0), ("2018-02-01T00:00:00Z", 1), ("2018-03-01T00:00:00Z", 0)],
)
def test_acceptance_window_boundaries(clock, count):
    payload = root()
    append_row(payload["filings"]["recent"], clock=clock)
    assert len(parse(payload).references) == count


@pytest.mark.parametrize("field", FIELDS)
def test_missing_parallel_field_refused(field):
    payload = root()
    del payload["filings"]["recent"][field]
    with pytest.raises(MetadataRefusal):
        parse(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("items", "bad"),
        ("accessionNumber", "bad"),
        ("acceptanceDateTime", "2018-02-01T00:00:00"),
        ("filingDate", "2018-02-30"),
        ("form", 8),
    ],
)
def test_invalid_native_scalars_refused_even_outside_window(field, value):
    payload = root()
    append_row(payload["filings"]["recent"], clock="2018-01-01T00:00:00Z")
    payload["filings"]["recent"][field][0] = value
    with pytest.raises(MetadataRefusal):
        parse(payload)


def test_parallel_length_and_duplicate_accession_refused():
    payload = root()
    payload["filings"]["recent"]["form"] = ["8-K"]
    with pytest.raises(MetadataRefusal):
        parse(payload)
    payload = root()
    append_row(payload["filings"]["recent"])
    append_row(payload["filings"]["recent"])
    with pytest.raises(MetadataRefusal):
        parse(payload)


@pytest.mark.parametrize("form", ["4", "4/A"])
def test_excluded_ownership_renderer_path_does_not_construct_document_url(form):
    payload = root()
    append_row(payload["filings"]["recent"], form=form)
    payload["filings"]["recent"]["primaryDocument"] = ["xslF345X06/form4.xml"]
    result = parse(payload)
    assert result.references == ()
    assert result.accessions == ("0000000001-18-000001",)


@pytest.mark.parametrize("path", ["../escape.htm", "xslF345X06/form4.xml"])
def test_selected_document_path_restriction_unchanged(path):
    payload = root()
    append_row(payload["filings"]["recent"])
    payload["filings"]["recent"]["primaryDocument"] = [path]
    with pytest.raises(MetadataRefusal):
        parse(payload)


@pytest.mark.parametrize("cik", [2, True, "0000000002", "bad"])
def test_root_identity_mismatch_refused(cik):
    payload = root()
    payload["cik"] = cik
    with pytest.raises(MetadataRefusal):
        parse(payload)


def test_flat_archive_requires_advertised_descriptor_and_retains_binding():
    payload = root()
    payload["filings"]["files"] = [descriptor()]
    selected = select_archive(parse(payload), start=START.date(), end_exclusive=END.date())
    flat = arrays()
    append_row(flat)
    result = parse_archive_metadata(
        flat, root=parse(payload), archive=selected, start=START, end_exclusive=END, forms=FORMS
    )
    assert result.issuer_name is None
    assert result.references[0].metadata_url.endswith(selected.name)
    with pytest.raises(MetadataRefusal):
        parse(flat)
    with pytest.raises(MetadataRefusal):
        parse_archive_metadata(
            payload,
            root=parse(payload),
            archive=selected,
            start=START,
            end_exclusive=END,
            forms=FORMS,
        )
    with pytest.raises(MetadataRefusal):
        parse_archive_metadata(
            flat, root=parse(root()), archive=selected, start=START, end_exclusive=END, forms=FORMS
        )


def test_archive_overlap_none_unique_and_ambiguous():
    assert select_archive(parse(root()), start=START.date(), end_exclusive=END.date()) is None
    payload = root()
    payload["filings"]["files"] = [descriptor(start="2018-03-01", end="2018-03-31")]
    assert select_archive(parse(payload), start=START.date(), end_exclusive=END.date()) is None
    payload["filings"]["files"] += [descriptor(2), descriptor(3)]
    with pytest.raises(MetadataRefusal):
        select_archive(parse(payload), start=START.date(), end_exclusive=END.date())


@pytest.mark.parametrize(
    "change",
    [
        {"name": "../escape.json"},
        {"name": "CIK0000000002-submissions-001.json"},
        {"filingCount": True},
        {"filingFrom": "2019-01-01"},
    ],
)
def test_archive_descriptor_refusals(change):
    payload = root()
    payload["filings"]["files"] = [descriptor() | change]
    with pytest.raises(MetadataRefusal):
        parse(payload)


def test_global_duplicate_refusal_and_aggregate_cap():
    payload = root()
    for sequence in range(1, 17):
        append_row(payload["filings"]["recent"], sequence=sequence)
    batch = parse(payload)
    assert len(combine_metadata((batch,), max_references=16)) == 16
    with pytest.raises(MetadataRefusal):
        combine_metadata((batch, batch), max_references=16)
    append_row(payload["filings"]["recent"], sequence=17)
    with pytest.raises(MetadataRefusal):
        combine_metadata((parse(payload),), max_references=16)


def test_exact_current_ticker_lead_preserves_native_title():
    payload = {"0": {"ticker": "BKNG", "cik_str": 1, "title": "OPAQUE METADATA CONTROL"}}
    result = select_ticker_lead(payload, ticker="BKNG")
    assert result.cik == CIK
    assert result.title == "OPAQUE METADATA CONTROL"
    assert result.ticker == "BKNG"
    with pytest.raises(MetadataRefusal):
        select_ticker_lead(payload, ticker="bkng")
    payload["1"] = payload["0"].copy()
    with pytest.raises(MetadataRefusal):
        select_ticker_lead(payload, ticker="BKNG")


@pytest.mark.parametrize("cik", [0, True, "1", -1, 10**10])
def test_invalid_current_cik_refused(cik):
    with pytest.raises(MetadataRefusal):
        select_ticker_lead(
            {"0": {"ticker": "BKNG", "cik_str": cik, "title": "OPAQUE METADATA CONTROL"}},
            ticker="BKNG",
        )


@respx.mock
def test_single_archive_capture_has_no_root_or_automatic_traversal():
    name = f"CIK{CIK}-submissions-001.json"
    route = respx.get(f"https://data.sec.gov/submissions/{name}").mock(
        return_value=httpx.Response(200, content=b"{}")
    )
    provider = SECEdgarProvider(user_agent="opaque test@example.com", min_interval_seconds=0)
    assert provider.capture_submissions_archive(CIK, name) == b"{}"
    assert route.call_count == 1
    assert len(respx.calls) == 1


@pytest.mark.parametrize(
    "cik,name",
    [
        (CIK, "../escape"),
        ("1", f"CIK{CIK}-submissions-001.json"),
        (CIK, "CIK0000000002-submissions-001.json"),
    ],
)
@respx.mock
def test_archive_capture_refuses_invalid_route_before_request(cik, name):
    with pytest.raises(ValueError):
        SECEdgarProvider(user_agent="opaque test@example.com").capture_submissions_archive(
            cik, name
        )
    assert len(respx.calls) == 0

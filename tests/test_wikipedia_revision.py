import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Final, TypeAlias

import pytest
from pydantic import JsonValue

from liq.data.wikipedia_revision import WikipediaRevisionRefused, parse_revision_metadata

CUTOFF: Final = datetime(2023, 4, 21, tzinfo=UTC)
JsonObject: TypeAlias = dict[str, JsonValue]


def opaque_metadata() -> tuple[JsonObject, JsonObject, JsonObject, JsonObject]:
    revision: JsonObject = {
        "revid": 202,
        "parentid": 201,
        "timestamp": "2023-04-20T00:00:00Z",
        "size": 123,
        "sha1": "a" * 40,
    }
    page: JsonObject = {
        "pageid": 101,
        "ns": 0,
        "title": "List of S&P 500 companies",
        "revisions": [revision],
    }
    query: JsonObject = {"pages": [page]}
    return {"batchcomplete": True, "query": query}, query, page, revision


def encoded(payload: JsonObject) -> bytes:
    return json.dumps(payload).encode()


def test_projects_only_a_scalar_locator_and_preserves_provider_literals() -> None:
    payload, _, _, revision = opaque_metadata()
    revision["sha1"] = "Ab" * 20
    payload["continue"] = {"rvcontinue": "opaque-token", "continue": "||"}
    result = parse_revision_metadata(encoded(payload), cutoff=CUTOFF)
    assert (result.page_id, result.namespace, result.title) == (
        101,
        0,
        "List of S&P 500 companies",
    )
    assert (result.revision_id, result.parent_revision_id) == (202, 201)
    assert result.revision_timestamp_literal == "2023-04-20T00:00:00Z"
    assert result.revision_timestamp_utc == datetime(2023, 4, 20, tzinfo=UTC)
    assert result.declared_content_bytes == 123
    assert result.provider_declared_sha1 == "Ab" * 20
    assert result.continuation_present is True
    assert result.content_verified is False


def test_exact_cutoff_and_zero_parent_or_size_are_metadata_not_content_claims() -> None:
    payload, _, _, revision = opaque_metadata()
    revision.update(timestamp="2023-04-21T00:00:00Z", parentid=0, size=0)
    result = parse_revision_metadata(encoded(payload), cutoff=CUTOFF)
    assert result.revision_timestamp_utc == CUTOFF
    assert result.parent_revision_id == result.declared_content_bytes == 0
    assert result.continuation_present is False


@pytest.mark.parametrize("value", ["a" * 39, "a" * 41, "z" * 31, "Ｆ" * 40, ""])
def test_unsupported_sha1_refuses_without_leaking_literal(value: str) -> None:
    payload, _, _, revision = opaque_metadata()
    revision["sha1"] = value
    with pytest.raises(WikipediaRevisionRefused) as caught:
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)
    assert str(caught.value) == caught.value.reason == "UNSUPPORTED_SHA1_REPRESENTATION"


@pytest.mark.parametrize(
    "changes",
    [
        {"revid": 0},
        {"revid": True},
        {"parentid": -1},
        {"size": -1},
        {"size": 1.5},
        {"size": "1"},
        {"sha1": None},
        {"timestamp": None},
    ],
)
def test_invalid_required_revision_metadata_refuses(changes: JsonObject) -> None:
    payload, _, _, revision = opaque_metadata()
    revision.update(changes)
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize(
    "key", ["sha1hidden", "texthidden", "suppressed", "content", "user", "comment"]
)
def test_unrequested_revision_fields_are_not_silently_dropped(key: str) -> None:
    payload, _, _, revision = opaque_metadata()
    revision[key] = "must-not-project"
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize("key", ["pageid", "ns", "title", "revisions"])
def test_missing_required_page_metadata_refuses(key: str) -> None:
    payload, _, page, _ = opaque_metadata()
    del page[key]
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize(
    "changes",
    [
        {"pageid": 0},
        {"ns": False},
        {"ns": 1},
        {"title": "Other"},
        {"missing": True},
    ],
)
def test_page_identity_must_match_the_requested_single_article(changes: JsonObject) -> None:
    payload, _, page, _ = opaque_metadata()
    page.update(changes)
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize("key", ["normalized", "redirects", "interwiki"])
def test_source_routing_changes_refuse(key: str) -> None:
    payload, query, _, _ = opaque_metadata()
    query[key] = []
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize("location", ["pages", "revisions"])
@pytest.mark.parametrize("copies", [0, 2])
def test_missing_or_ambiguous_metadata_refuses(location: str, copies: int) -> None:
    payload, query, page, revision = opaque_metadata()
    if location == "pages":
        pages: list[JsonValue] = [page]
        query["pages"] = pages * copies
    else:
        revisions: list[JsonValue] = [revision]
        page["revisions"] = revisions * copies
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize(
    "value",
    [
        "2023-04-22T00:00:00Z",
        "2023-04-20T00:00:00+00:00",
        "2023-02-30T00:00:00Z",
    ],
)
def test_provider_clock_must_be_literal_utc_and_not_later_than_cutoff(value: str) -> None:
    payload, _, _, revision = opaque_metadata()
    revision["timestamp"] = value
    with pytest.raises(WikipediaRevisionRefused, match="REVISION"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize(
    "cutoff",
    [
        datetime(2023, 4, 21),
        CUTOFF.replace(microsecond=1),
        CUTOFF.astimezone(timezone(timedelta(hours=1))),
    ],
)
def test_ambiguous_cutoff_refuses_before_decoding(cutoff: datetime) -> None:
    with pytest.raises(WikipediaRevisionRefused, match="INVALID_CUTOFF"):
        parse_revision_metadata(b"not-json", cutoff=cutoff)


@pytest.mark.parametrize("raw", [b"", b"x" * (64 * 1024)])
def test_response_byte_budget_is_exclusive(raw: bytes) -> None:
    with pytest.raises(WikipediaRevisionRefused, match="BODY_BYTE_BUDGET"):
        parse_revision_metadata(raw, cutoff=CUTOFF)


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        b"[]",
        b"{",
        b'{"error":{"code":"opaque"}}',
        b'{"query":null}',
        b"[" * 2000,
    ],
)
def test_invalid_response_never_leaks_body_text(raw: bytes) -> None:
    with pytest.raises(WikipediaRevisionRefused) as caught:
        parse_revision_metadata(raw, cutoff=CUTOFF)
    assert str(caught.value) == caught.value.reason == "METADATA_SCHEMA"


def test_duplicate_keys_refuse_instead_of_last_value_winning() -> None:
    raw = encoded(opaque_metadata()[0]).replace(b'"revid": 202', b'"revid": 202, "revid": 203')
    with pytest.raises(WikipediaRevisionRefused, match="DUPLICATE_JSON_KEY"):
        parse_revision_metadata(raw, cutoff=CUTOFF)


def test_nonfinite_json_refuses_before_metadata_projection() -> None:
    raw = encoded(opaque_metadata()[0]).replace(b'"size": 123', b'"size": NaN')
    with pytest.raises(WikipediaRevisionRefused, match="NONFINITE_JSON"):
        parse_revision_metadata(raw, cutoff=CUTOFF)


@pytest.mark.parametrize(
    "continuation",
    [
        {},
        {"rvcontinue": "x"},
        {"rvcontinue": "x", "continue": "||", "other": "x"},
        {"rvcontinue": "x" * 513, "continue": "||"},
        {"rvcontinue": "Ｆ", "continue": "||"},
    ],
)
def test_unexpected_or_unbounded_continuation_refuses(continuation: JsonObject) -> None:
    payload = opaque_metadata()[0]
    payload["continue"] = continuation
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)


@pytest.mark.parametrize("field", ["continue", "batchcomplete"])
def test_explicit_null_optional_field_is_not_absence(field: str) -> None:
    payload = opaque_metadata()[0]
    payload[field] = None
    with pytest.raises(WikipediaRevisionRefused, match="METADATA_SCHEMA"):
        parse_revision_metadata(encoded(payload), cutoff=CUTOFF)

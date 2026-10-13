"""Bounded native SEC metadata selection, without event or continuity inference."""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime

from liq.data.providers.sec_edgar import _parse_acceptance, sec_archive_document_url

_FIELDS = (
    "form",
    "accessionNumber",
    "acceptanceDateTime",
    "filingDate",
    "items",
    "primaryDocument",
)
_MAX_ROWS = 20_000
_MAX_ARCHIVES = 1_000
_MAX_STRING = 4_096


class MetadataRefusal(ValueError):
    """Native metadata cannot support an unambiguous bounded selection."""


@dataclass(frozen=True)
class TickerLead:
    cik: str
    ticker: str
    title: str


@dataclass(frozen=True)
class SubmissionArchive:
    name: str
    filing_count: int
    filing_from: date
    filing_to: date


@dataclass(frozen=True)
class FilingMetadata:
    cik: str
    accession: str
    form: str
    primary_document: str
    filing_date: date
    acceptance_literal: str
    acceptance_utc: datetime
    items: tuple[str, ...]
    metadata_url: str
    document_url: str
    native_items: str
    item_whitespace_count: int
    item_empty_segments: int


@dataclass(frozen=True)
class ItemLexicon:
    """Parsed disclosure codes with the authoritative native text preserved."""

    native_literal: str
    tokens: tuple[str, ...]
    whitespace_count: int
    empty_segments: int
    interior_whitespace: bool
    unsupported_grammar: bool


@dataclass(frozen=True)
class MetadataBatch:
    cik: str
    issuer_name: str | None
    references: tuple[FilingMetadata, ...]
    archives: tuple[SubmissionArchive, ...]
    accessions: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveDescriptorBatch:
    """Validated current-root identity and archive advertisements; no filing rows."""

    cik: str
    issuer_name: str
    archives: tuple[SubmissionArchive, ...]


@dataclass(frozen=True)
class FilingLocator:
    """Native filing pointer; disclosure item semantics are unassessed."""

    cik: str
    accession: str
    form: str
    primary_document: str
    filing_date: date
    acceptance_literal: str
    acceptance_utc: datetime
    metadata_url: str
    document_url: str


@dataclass(frozen=True)
class FilingLocatorBatch:
    """Current root identity and bounded pointers; not admitted full metadata."""

    cik: str
    issuer_name: str
    references: tuple[FilingLocator, ...]


def _text(value: object, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > _MAX_STRING or (not empty and not value.strip()):
        raise MetadataRefusal("missing, invalid or oversized native string")
    return value


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MetadataRefusal("native JSON object required")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise MetadataRefusal("native string object keys required")
        result[key] = item
    return result


def tokenize_native_items(value: object) -> ItemLexicon:
    """Handle bounded ASCII edge whitespace/empty separators; never repair tokens."""
    literal = _text(value, empty=True)
    tokens: list[str] = []
    whitespace_count = 0
    empty_segments = 0
    if literal:
        for segment in literal.split(","):
            token = segment.strip(" \t\r\n\v\f")
            whitespace_count += len(segment) - len(token)
            if token:
                tokens.append(token)
            else:
                empty_segments += 1
    return ItemLexicon(
        literal,
        tuple(tokens),
        whitespace_count,
        empty_segments,
        any(any(character in " \t\r\n\v\f" for character in token) for token in tokens),
        any(re.fullmatch(r"[0-9]+\.[0-9]{2}", token) is None for token in tokens),
    )


def _cik(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{10}", value) is None or int(value) == 0:
        raise MetadataRefusal("canonical nonzero ten-digit CIK required")
    return value


def _native_cik(value: object) -> str:
    if type(value) is int and 0 < value < 10**10:
        return f"{value:010d}"
    return _cik(value)


def _date(value: object) -> date:
    text = _text(value)
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text) is None:
        raise MetadataRefusal("invalid native date")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise MetadataRefusal("invalid native date") from exc


def validate_archive_name(cik: str, name: str) -> str:
    """Validate one issuer-bound basename; advertisement is checked separately."""
    _cik(cik)
    if (
        not isinstance(name, str)
        or re.fullmatch(rf"CIK{cik}-submissions-[0-9]{{3}}\.json", name) is None
    ):
        raise MetadataRefusal("invalid issuer-bound archive name")
    return name


def select_ticker_lead(payload: object, *, ticker: str) -> TickerLead:
    """Require exactly one native literal ticker; this is a current locator only."""
    _text(ticker)
    data = _object(payload)
    if len(data) > 100_000:
        raise MetadataRefusal("invalid or oversized current ticker object")
    matches: list[TickerLead] = []
    for raw in data.values():
        row = _object(raw)
        native_ticker = _text(row.get("ticker"))
        if native_ticker == ticker:
            matches.append(
                TickerLead(_native_cik(row.get("cik_str")), native_ticker, _text(row.get("title")))
            )
    if len(matches) != 1:
        raise MetadataRefusal("current ticker must have exactly one native match")
    return matches[0]


def _window(start: datetime, end_exclusive: datetime, forms: frozenset[str]) -> None:
    if start.utcoffset() is None or end_exclusive.utcoffset() is None or start >= end_exclusive:
        raise MetadataRefusal("explicit timezone and increasing window required")
    if (
        not forms
        or len(forms) > 32
        or any(not isinstance(form, str) or len(form) > 32 for form in forms)
    ):
        raise MetadataRefusal("invalid bounded form selection")


def _parallel_columns(payload: object, fields: tuple[str, ...]) -> list[list[str]]:
    data = _object(payload)
    columns: list[list[str]] = []
    for field in fields:
        column = data.get(field)
        if not isinstance(column, list) or len(column) > _MAX_ROWS:
            raise MetadataRefusal("missing or oversized parallel array")
        columns.append([_text(value, empty=field == "items") for value in column])
    if len({len(column) for column in columns}) != 1:
        raise MetadataRefusal("parallel array length mismatch")
    return columns


def _check_accession(accession: str, seen: set[str]) -> None:
    if re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession) is None or accession in seen:
        raise MetadataRefusal("invalid or duplicate accession")
    seen.add(accession)


def _acceptance_utc(clock: str) -> datetime:
    try:
        return _parse_acceptance(clock).astimezone(UTC)
    except ValueError as exc:
        raise MetadataRefusal("invalid acceptance clock") from exc


def _document_url(cik: str, accession: str, document: str) -> str:
    try:
        return sec_archive_document_url(cik, accession, document)
    except ValueError as exc:
        raise MetadataRefusal("invalid selected document path") from exc


def _rows(
    payload: object,
    *,
    cik: str,
    source_url: str,
    start: datetime,
    end_exclusive: datetime,
    forms: frozenset[str],
) -> tuple[tuple[FilingMetadata, ...], tuple[str, ...]]:
    columns = _parallel_columns(payload, _FIELDS)
    selected: list[FilingMetadata] = []
    accessions: list[str] = []
    seen: set[str] = set()
    for form, accession, clock, filed, items, document in zip(*columns, strict=True):
        _check_accession(accession, seen)
        accessions.append(accession)
        lexical = tokenize_native_items(items)
        if lexical.unsupported_grammar:
            raise MetadataRefusal("invalid native item token")
        accepted = _acceptance_utc(clock)
        filing_date = _date(filed)
        if form in forms and start <= accepted < end_exclusive:
            url = _document_url(cik, accession, document)
            selected.append(
                FilingMetadata(
                    cik,
                    accession,
                    form,
                    document,
                    filing_date,
                    clock,
                    accepted,
                    lexical.tokens,
                    source_url,
                    url,
                    lexical.native_literal,
                    lexical.whitespace_count,
                    lexical.empty_segments,
                )
            )
    return tuple(selected), tuple(accessions)


def parse_root_metadata(
    payload: object, *, cik: str, start: datetime, end_exclusive: datetime, forms: frozenset[str]
) -> MetadataBatch:
    """Select root references and validate its advertised archive descriptors."""
    _cik(cik)
    _window(start, end_exclusive, forms)
    name, filings = _root_identity(payload, cik=cik)
    references, accessions = _rows(
        filings.get("recent"),
        cik=cik,
        source_url=f"https://data.sec.gov/submissions/CIK{cik}.json",
        start=start,
        end_exclusive=end_exclusive,
        forms=forms,
    )
    archives = _archive_descriptors(filings.get("files"), cik=cik)
    return MetadataBatch(cik, name, references, archives, accessions)


def _root_identity(payload: object, *, cik: str) -> tuple[str, dict[str, object]]:
    _cik(cik)
    data = _object(payload)
    if _native_cik(data.get("cik")) != cik:
        raise MetadataRefusal("root issuer mismatch or unsupported shape")
    return _text(data.get("name")), _object(data.get("filings"))


def _archive_descriptors(descriptors: object, *, cik: str) -> tuple[SubmissionArchive, ...]:
    if not isinstance(descriptors, list) or len(descriptors) > _MAX_ARCHIVES:
        raise MetadataRefusal("invalid or oversized archive descriptors")
    archives: list[SubmissionArchive] = []
    names: set[str] = set()
    for raw in descriptors:
        row = _object(raw)
        filename = validate_archive_name(cik, _text(row.get("name")))
        count = row.get("filingCount")
        first, last = _date(row.get("filingFrom")), _date(row.get("filingTo"))
        if filename in names or type(count) is not int or count <= 0 or first > last:
            raise MetadataRefusal("duplicate or invalid archive descriptor")
        names.add(filename)
        archives.append(SubmissionArchive(filename, count, first, last))
    return tuple(archives)


def parse_root_archive_descriptors(payload: object, *, cik: str) -> ArchiveDescriptorBatch:
    """Validate only root identity/files, never interpret or admit recent rows."""
    name, filings = _root_identity(payload, cik=cik)
    return ArchiveDescriptorBatch(cik, name, _archive_descriptors(filings.get("files"), cik=cik))


def parse_root_filing_locators(
    payload: object,
    *,
    cik: str,
    start: datetime,
    end_exclusive: datetime,
    forms: frozenset[str],
    maximum: int,
) -> FilingLocatorBatch:
    """Select bounded pointers without interpreting items or archive descriptors."""
    _window(start, end_exclusive, forms)
    if type(maximum) is not int or maximum <= 0:
        raise MetadataRefusal("positive reference ceiling required")
    name, filings = _root_identity(payload, cik=cik)
    fields = tuple(field for field in _FIELDS if field != "items")
    columns = _parallel_columns(filings.get("recent"), fields)
    seen: set[str] = set()
    references: list[FilingLocator] = []
    for form, accession, clock, filed, document in zip(*columns, strict=True):
        _check_accession(accession, seen)
        accepted = _acceptance_utc(clock)
        filing_date = _date(filed)
        if form in forms and start <= accepted < end_exclusive:
            references.append(
                FilingLocator(
                    cik,
                    accession,
                    form,
                    document,
                    filing_date,
                    clock,
                    accepted,
                    f"https://data.sec.gov/submissions/CIK{cik}.json",
                    _document_url(cik, accession, document),
                )
            )
            if len(references) > maximum:
                raise MetadataRefusal("aggregate reference budget exceeded")
    return FilingLocatorBatch(cik, name, tuple(references))


def select_archive(
    root: MetadataBatch | ArchiveDescriptorBatch, *, start: date, end_exclusive: date
) -> SubmissionArchive | None:
    """Require at most one filing-date overlap, without a coverage claim."""
    if start >= end_exclusive:
        raise MetadataRefusal("increasing archive locator window required")
    matches = tuple(
        item
        for item in root.archives
        if item.filing_from < end_exclusive and item.filing_to >= start
    )
    if len(matches) > 1:
        raise MetadataRefusal("ambiguous overlapping archives")
    return matches[0] if matches else None


def parse_archive_metadata(
    payload: object,
    *,
    root: MetadataBatch,
    archive: SubmissionArchive,
    start: datetime,
    end_exclusive: datetime,
    forms: frozenset[str],
) -> MetadataBatch:
    """Parse flat arrays only when bound to a descriptor from the held root."""
    _cik(root.cik)
    _window(start, end_exclusive, forms)
    validate_archive_name(root.cik, archive.name)
    if archive not in root.archives or not isinstance(payload, dict) or "filings" in payload:
        raise MetadataRefusal("unadvertised archive or unsupported archive shape")
    references, accessions = _rows(
        payload,
        cik=root.cik,
        source_url=f"https://data.sec.gov/submissions/{archive.name}",
        start=start,
        end_exclusive=end_exclusive,
        forms=forms,
    )
    return MetadataBatch(root.cik, None, references, (), accessions)


def combine_metadata(
    batches: tuple[MetadataBatch, ...], *, max_references: int
) -> tuple[FilingMetadata, ...]:
    """Refuse duplicate accessions or overflow; never deduplicate or truncate."""
    if type(max_references) is not int or max_references < 1 or not batches:
        raise MetadataRefusal("positive aggregate bound and batches required")
    seen: set[str] = set()
    selected: list[FilingMetadata] = []
    for batch in batches:
        if batch.cik != batches[0].cik or seen.intersection(batch.accessions):
            raise MetadataRefusal("issuer mismatch or cross-object duplicate accession")
        seen.update(batch.accessions)
        selected.extend(batch.references)
        if len(selected) > max_references:
            raise MetadataRefusal("aggregate reference budget exceeded")
    return tuple(selected)

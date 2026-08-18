"""Deterministic corporate-action evidence extraction from SEC filings.

This parser is intentionally conservative. It recognizes explicit stock-split
ratios/dates and cash-dividend amounts/dates in complete-submission documents.
It never derives an ex-date from a record or payment date and never treats an
empty parse as evidence that no corporate action occurred.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Any

from liq.data.providers.sec_edgar import (
    EdgarFilingDocument,
    complete_submission_url,
    parse_edgar_documents,
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
}
_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)"
)
_DATE = rf"(?:{_WEEKDAY},?\s+)?{_MONTH}\s+\d{{1,2}},\s+\d{{4}}"
_RATIO_TOKEN = (
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
)

_SPLIT_RE = re.compile(
    rf"\b(?P<to>{_RATIO_TOKEN})\s*[- ]\s*for\s*[- ]\s*"
    rf"(?P<from>{_RATIO_TOKEN})(?:\s+(?:forward|reverse))?\s+(?:stock\s+)?split\b",
    re.IGNORECASE,
)
_CASH_DIVIDEND_RE = re.compile(r"\b(?:quarterly\s+)?cash dividend\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s+per\s+share", re.IGNORECASE)
_RECORD_RE = re.compile(
    r"(?:shareholders?|stockholders?)\s+of\s+record|"
    r"record\s+holders?\s+of\s+common\s+stock",
    re.IGNORECASE,
)
_EX_DATE_RE = re.compile(
    rf"\bex[-\s]?dividend date\b.{{0,80}}?({_DATE})",
    re.IGNORECASE,
)
_PAYMENT_RE = re.compile(
    rf"(?:\bis\s+payable\b|\bwill\s+be\s+paid\b|\bpayable\b).{{0,140}}?({_DATE})",
    re.IGNORECASE,
)
_DISTRIBUTION_RE = re.compile(
    rf"\bdistributed\b.{{0,140}}?({_DATE})",
    re.IGNORECASE,
)
_EFFECTIVE_RE = re.compile(
    rf"\btrading\b.{{0,160}}?\b(?:begin|commence)\w*\b.{{0,160}}?"
    rf"\bsplit-adjusted basis\b.{{0,120}}?({_DATE})",
    re.IGNORECASE,
)

_ACTION_DATE_FIELDS = (
    "ex_date",
    "effective_date",
    "record_date",
    "payment_date",
    "distribution_date",
    "filing_date",
)

_MARKED_SECTION_START_RE = re.compile(r"<!\[\s*([A-Za-z][A-Za-z0-9:_-]*)")
NON_TEXT_DOCUMENT_TYPES = frozenset({"EXCEL", "GRAPHIC", "ZIP"})
PDF_TEXT_EXTRACTION_TIMEOUT_SECONDS = 30.0


class EdgarMarkupError(ValueError):
    """A filing contains markup that cannot be safely reduced to plain text."""


def extract_pdf_text(
    payload: bytes,
    *,
    timeout_seconds: float = PDF_TEXT_EXTRACTION_TIMEOUT_SECONDS,
) -> str:
    """Extract normalized text from a verified SEC PDF or fail closed.

    ``pdftotext`` is invoked with argument-vector input and an in-memory PDF;
    no SEC-controlled filename reaches the shell or local filesystem.
    """
    if not payload.startswith(b"%PDF-"):
        raise EdgarMarkupError("standalone SEC attachment does not start with PDF magic")
    executable = shutil.which("pdftotext")
    if executable is None:
        raise EdgarMarkupError("pdftotext executable is unavailable for SEC PDF extraction")
    try:
        completed = subprocess.run(
            [executable, "-layout", "-", "-"],
            input=payload,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise EdgarMarkupError("SEC PDF text extraction timed out") from exc
    if completed.returncode != 0:
        raise EdgarMarkupError(f"SEC PDF text extraction exited with code {completed.returncode}")
    text = " ".join(completed.stdout.decode("utf-8", errors="replace").split())
    if not text:
        raise EdgarMarkupError("SEC PDF text extraction produced no extractable text")
    return text


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


@dataclass(frozen=True, slots=True)
class EdgarCorporateAction:
    """One explicit filing disclosure with accession-bound provenance."""

    symbol: str
    cik: str
    filing_date: str
    acceptance_datetime: str
    source_accession: str
    source_url: str
    source_document_type: str
    source_document_filename: str | None
    corporate_action_type: str
    ex_date: str | None
    effective_date: str | None
    record_date: str | None
    payment_date: str | None
    distribution_date: str | None
    split_ratio_from: int | None
    split_ratio_to: int | None
    cash_amount: str | None
    currency: str | None
    parse_status: str
    evidence_sha256: str
    evidence_excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _plain_text(raw: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except AssertionError as exc:
        match = _MARKED_SECTION_START_RE.search(raw)
        marker = f"<![{match.group(1)}" if match else "unidentified marked section"
        raise EdgarMarkupError(f"unsupported EDGAR marked section {marker!r}") from exc
    return " ".join(" ".join(parser.parts).split())


def _number(raw: str) -> int:
    normalized = raw.casefold()
    return int(normalized) if normalized.isdigit() else _NUMBER_WORDS[normalized]


def _parse_date(raw: str) -> date:
    without_weekday = re.sub(rf"^{_WEEKDAY},?\s+", "", raw, flags=re.IGNORECASE)
    return datetime.strptime(without_weekday, "%B %d, %Y").date()


def _date_after(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return _parse_date(match.group(1)).isoformat() if match else None


def _record_date(text: str) -> str | None:
    for anchor in _RECORD_RE.finditer(text):
        tail = text[anchor.end() : anchor.end() + 240]
        dates = re.findall(_DATE, tail, flags=re.IGNORECASE)
        if dates:
            return _parse_date(dates[0]).isoformat()
    return None


def _excerpt(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _base_action(
    *,
    symbol: str,
    cik: str,
    filing_date: date,
    acceptance_datetime: datetime,
    accession_number: str,
    document_type: str,
    filename: str | None,
    document_text: str,
    excerpt: str,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "cik": str(cik).zfill(10),
        "filing_date": filing_date.isoformat(),
        "acceptance_datetime": acceptance_datetime.isoformat(),
        "source_accession": accession_number,
        "source_url": complete_submission_url(cik, accession_number),
        "source_document_type": document_type,
        "source_document_filename": filename,
        "evidence_sha256": hashlib.sha256(document_text.encode("utf-8")).hexdigest(),
        "evidence_excerpt": _excerpt(excerpt),
    }


def _split_action(
    match: re.Match[str],
    segment: str,
    base: dict[str, Any],
) -> EdgarCorporateAction:
    effective = _date_after(_EFFECTIVE_RE, segment)
    ratio_from = _number(match.group("from"))
    ratio_to = _number(match.group("to"))
    return EdgarCorporateAction(
        **base,
        corporate_action_type="stock_split",
        ex_date=None,
        effective_date=effective,
        record_date=_record_date(segment),
        payment_date=None,
        distribution_date=_date_after(_DISTRIBUTION_RE, segment),
        split_ratio_from=ratio_from,
        split_ratio_to=ratio_to,
        cash_amount=None,
        currency=None,
        parse_status=(
            "EXPLICIT_EFFECTIVE_DATE_AND_RATIO"
            if effective is not None
            else "MISSING_EFFECTIVE_DATE"
        ),
    )


def _dividend_action(segment: str, base: dict[str, Any]) -> EdgarCorporateAction:
    amounts = _MONEY_RE.findall(segment)
    ex_date = _date_after(_EX_DATE_RE, segment)
    return EdgarCorporateAction(
        **base,
        corporate_action_type="cash_dividend",
        ex_date=ex_date,
        effective_date=None,
        record_date=_record_date(segment),
        payment_date=_date_after(_PAYMENT_RE, segment),
        distribution_date=None,
        split_ratio_from=None,
        split_ratio_to=None,
        cash_amount=amounts[-1] if amounts else None,
        currency="USD" if amounts else None,
        parse_status="EXPLICIT_EX_DATE" if ex_date is not None else "MISSING_EX_DATE",
    )


def _richness(action: EdgarCorporateAction) -> int:
    return sum(
        value is not None
        for value in (
            action.ex_date,
            action.effective_date,
            action.record_date,
            action.payment_date,
            action.distribution_date,
            action.split_ratio_from,
            action.split_ratio_to,
            action.cash_amount,
        )
    )


def parse_edgar_corporate_actions(
    submission_text: str,
    *,
    symbol: str,
    cik: str,
    filing_date: date,
    acceptance_datetime: datetime,
    accession_number: str,
    pdf_text_resolver: Callable[[EdgarFilingDocument], str] | None = None,
) -> list[dict[str, Any]]:
    """Extract explicit action facts from one complete-submission payload."""
    candidates: list[EdgarCorporateAction] = []
    for document in parse_edgar_documents(submission_text):
        if document.document_type in NON_TEXT_DOCUMENT_TYPES:
            continue
        try:
            is_pdf = document.document_type == "PDF" or bool(
                document.filename and document.filename.casefold().endswith(".pdf")
            )
            if is_pdf and pdf_text_resolver is not None:
                text = " ".join(pdf_text_resolver(document).split())
                if not text:
                    raise EdgarMarkupError("SEC PDF resolver produced no extractable text")
            else:
                text = _plain_text(document.text)
                if is_pdf:
                    raise EdgarMarkupError("action-bearing PDF attachment requires text extraction")
        except EdgarMarkupError as exc:
            filename = f", filename {document.filename!r}" if document.filename else ""
            raise EdgarMarkupError(
                f"document type {document.document_type!s}{filename}: {exc}"
            ) from exc
        anchors: list[tuple[int, str, re.Match[str]]] = [
            (match.start(), "stock_split", match) for match in _SPLIT_RE.finditer(text)
        ]
        anchors.extend(
            (match.start(), "cash_dividend", match) for match in _CASH_DIVIDEND_RE.finditer(text)
        )
        anchors.sort(key=lambda item: item[0])
        for index, (start, action_type, match) in enumerate(anchors):
            end = anchors[index + 1][0] if index + 1 < len(anchors) else len(text)
            segment = text[start:end]
            base = _base_action(
                symbol=symbol,
                cik=cik,
                filing_date=filing_date,
                acceptance_datetime=acceptance_datetime,
                accession_number=accession_number,
                document_type=document.document_type,
                filename=document.filename,
                document_text=text,
                excerpt=segment,
            )
            if action_type == "stock_split":
                candidates.append(_split_action(match, segment, base))
            else:
                candidates.append(_dividend_action(segment, base))

    dated_core_keys = {
        (
            action.corporate_action_type,
            action.split_ratio_from,
            action.split_ratio_to,
            action.cash_amount,
        )
        for action in candidates
        if any(
            value is not None
            for value in (
                action.ex_date,
                action.effective_date,
                action.record_date,
                action.payment_date,
                action.distribution_date,
            )
        )
    }
    deduped: dict[tuple[object, ...], EdgarCorporateAction] = {}
    for action in candidates:
        core_key = (
            action.corporate_action_type,
            action.split_ratio_from,
            action.split_ratio_to,
            action.cash_amount,
        )
        date_key = (
            action.ex_date,
            action.effective_date,
            action.record_date,
            action.payment_date,
            action.distribution_date,
        )
        if core_key in dated_core_keys and all(value is None for value in date_key):
            continue
        key = (*core_key, *date_key)
        prior = deduped.get(key)
        if prior is None or _richness(action) > _richness(prior):
            deduped[key] = action
    return [
        action.to_dict()
        for action in sorted(
            deduped.values(),
            key=lambda row: (
                row.corporate_action_type,
                row.effective_date or row.record_date or "",
            ),
        )
    ]


def action_range_match_basis(
    action: dict[str, Any],
    *,
    start: date,
    end: date,
) -> str | None:
    """Return the first explicit action date inside an inclusive query range."""
    for field in _ACTION_DATE_FIELDS:
        raw = action.get(field)
        if raw is None:
            continue
        observed = date.fromisoformat(str(raw))
        if start <= observed <= end:
            return field
    return None


__all__ = [
    "EdgarCorporateAction",
    "EdgarMarkupError",
    "NON_TEXT_DOCUMENT_TYPES",
    "action_range_match_basis",
    "extract_pdf_text",
    "parse_edgar_corporate_actions",
]

"""SEC EDGAR filings provider (reference/event data, not OHLCV).

Fetches registered 8-K item-family filing metadata per symbol from the free
SEC EDGAR endpoints, using ``acceptanceDateTime`` as the public-dissemination
proxy. The narrower Item 2.02 earnings-event API remains available for
backward compatibility.

Deliberately NOT a :class:`~liq.data.providers.base.BaseProvider` subclass:
that ABC's contract is OHLCV bars (``fetch_bars``/``list_instruments``), which
an event provider cannot honestly implement (interface segregation). This is
the stack's first event/reference adapter and defines its own narrow surface.

SEC fair-use etiquette: a real contact ``User-Agent`` is required (no default),
and requests are throttled to 8/sec via the shared :class:`RateLimiter`.
"""

from __future__ import annotations

import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Collection
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

import httpx
import polars as pl

from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.rate_limiter import RateLimiter

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_BROWSE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SUBMISSIONS_ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"
COMPLETE_SUBMISSION_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}.txt"
)
SEC_ARCHIVE_DOCUMENT_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
)

_ACCESSION_PATTERN = re.compile(r"\d{10}-\d{2}-\d{6}")
_DOCUMENT_PATTERN = re.compile(r"<DOCUMENT>\s*(.*?)</DOCUMENT>", re.IGNORECASE | re.DOTALL)

_EVENT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "symbol": pl.String,
    "cik": pl.String,
    "filing_date": pl.Date,
    "acceptance_datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    "accession_number": pl.String,
    "items": pl.String,
}

DEFAULT_8K_ITEMS = frozenset({"1.01", "2.02", "5.02", "7.01", "8.01"})
CORPORATE_ACTION_8K_ITEMS = frozenset({"2.02", "3.03", "5.03", "7.01", "8.01"})
CORPORATE_ACTION_FILING_LOOKBACK_DAYS = 370

_GENERIC_8K_EVENT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    **_EVENT_SCHEMA,
    "matched_items": pl.List(pl.String),
    "primary_document": pl.String,
    "primary_document_description": pl.String,
}


@dataclass(frozen=True)
class EdgarAccessionMetadata:
    """Accession-bound identity and attachment metadata from SEC SGML."""

    cik: str
    accession_number: str
    filing_symbols: tuple[str, ...]
    has_ex_99_1: bool
    document_types: tuple[str, ...]
    source_url: str


@dataclass(frozen=True)
class EdgarTickerCandidate:
    """One current SEC ticker/CIK/name association, without PIT semantics."""

    cik: str
    ticker: str
    title: str


@dataclass(frozen=True)
class EdgarTickerDiscoveryCandidate:
    """SEC company-browse candidate requiring accession confirmation."""

    cik: str
    queried_ticker: str
    title: str
    source_url: str


@dataclass(frozen=True)
class EdgarFilingIndexEntry:
    """One accession reference from the official SEC submissions index."""

    cik: str
    filing_date: date
    accession_number: str
    form: str
    source_url: str


@dataclass(frozen=True)
class EdgarFilingClockEntry:
    """One accession with its official SEC acceptance clock."""

    cik: str
    filing_date: date
    acceptance_datetime: datetime
    accession_number: str
    form: str
    source_url: str


@dataclass(frozen=True)
class EdgarFilingDocument:
    """One document from an EDGAR complete-submission SGML envelope."""

    document_type: str
    filename: str | None
    description: str | None
    text: str


class _TradingSymbolParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture_depth = 0
        self._parts: list[str] = []
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._capture_depth:
            self._capture_depth += 1
            return
        attributes = {name.casefold(): value for name, value in attrs}
        fact_name = attributes.get("name")
        if (
            tag.casefold() == "ix:nonnumeric"
            and fact_name
            and fact_name.casefold() == "dei:tradingsymbol"
        ):
            self._capture_depth = 1
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        del tag
        if not self._capture_depth:
            return
        self._capture_depth -= 1
        if self._capture_depth == 0:
            self.values.append("".join(self._parts))
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture_depth:
            self._parts.append(data)


def complete_submission_url(cik: int | str, accession_number: str) -> str:
    """Return the official complete-submission URL for one accession."""
    cik10 = _pad_cik(cik)
    if len(cik10) != 10 or not cik10.isdigit():
        raise ValueError("CIK must contain at most 10 decimal digits")
    if _ACCESSION_PATTERN.fullmatch(accession_number) is None:
        raise ValueError("accession_number must match ##########-##-######")
    return COMPLETE_SUBMISSION_URL.format(
        cik=int(cik10),
        accession_nodash=accession_number.replace("-", ""),
        accession=accession_number,
    )


def sec_archive_document_url(
    cik: int | str,
    accession_number: str,
    filename: str,
) -> str:
    """Return the official URL for a safe accession attachment basename."""
    cik10 = _pad_cik(cik)
    if len(cik10) != 10 or not cik10.isdigit():
        raise ValueError("CIK must contain at most 10 decimal digits")
    if _ACCESSION_PATTERN.fullmatch(accession_number) is None:
        raise ValueError("accession_number must match ##########-##-######")
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("SEC document filename must be a basename")
    return SEC_ARCHIVE_DOCUMENT_URL.format(
        cik=int(cik10),
        accession_nodash=accession_number.replace("-", ""),
        filename=quote(filename, safe="-._"),
    )


def _sgml_field(document: str, field: str) -> str | None:
    match = re.search(rf"(?im)^\s*<{field}>\s*([^\r\n<]+)", document)
    return match.group(1).strip() if match else None


def _sgml_text(document: str) -> str:
    match = re.search(r"<TEXT>\s*(.*?)</TEXT>", document, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def parse_edgar_documents(submission_text: str) -> tuple[EdgarFilingDocument, ...]:
    """Parse the typed documents in one complete-submission SGML payload."""
    documents: list[EdgarFilingDocument] = []
    for document in _DOCUMENT_PATTERN.findall(submission_text):
        document_type = _sgml_field(document, "TYPE")
        if document_type is None:
            continue
        documents.append(
            EdgarFilingDocument(
                document_type=document_type.upper(),
                filename=_sgml_field(document, "FILENAME"),
                description=_sgml_field(document, "DESCRIPTION"),
                text=_sgml_text(document),
            )
        )
    return tuple(documents)


def _filing_symbols(primary_documents: list[str]) -> tuple[str, ...]:
    symbols: list[str] = []
    seen: set[str] = set()
    for document in primary_documents:
        parser = _TradingSymbolParser()
        # Some legacy SEC filings contain malformed marked sections after the
        # Inline XBRL cover facts. Keep facts already parsed; never infer or
        # backfill a missing symbol.
        with suppress(AssertionError):
            parser.feed(document)
        for raw_value in parser.values:
            symbol = " ".join(raw_value.split()).upper()
            if not symbol or not any(character.isalnum() for character in symbol):
                continue
            if symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
    return tuple(symbols)


def parse_edgar_accession_metadata(
    submission_text: str,
    *,
    cik: int | str,
    accession_number: str,
) -> EdgarAccessionMetadata:
    """Parse filing-date symbols and the EX-99.1 flag without market data."""
    source_url = complete_submission_url(cik, accession_number)
    cik10 = _pad_cik(cik)
    document_types: list[str] = []
    primary_documents: list[str] = []
    for document in parse_edgar_documents(submission_text):
        document_types.append(document.document_type)
        if document.document_type == "8-K":
            primary_documents.append(document.text)
    return EdgarAccessionMetadata(
        cik=cik10,
        accession_number=accession_number,
        filing_symbols=_filing_symbols(primary_documents),
        has_ex_99_1="EX-99.1" in document_types,
        document_types=tuple(document_types),
        source_url=source_url,
    )


def raw_form4_url(cik: int | str, accession: str, primary_document: str) -> str:
    """Archive URL for the raw ownershipDocument XML.

    Submissions metadata often lists the XSL-rendered path
    (``xslF345X06/form4.xml``) as the primary document; the raw XML lives at
    the same basename without the renderer prefix.
    """
    accession_nodash = accession.replace("-", "")
    document = primary_document.rsplit("/", 1)[-1]
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{document}"


def _element_value(node: Any) -> str | None:
    value = node.find("value") if node is not None else None
    return value.text.strip() if value is not None and value.text else None


def _flag(root: Any, path: str) -> bool:
    node = root.find(path)
    return node is not None and (node.text or "").strip().lower() in ("1", "true")


def parse_form4_purchases(xml_text: str) -> list[dict[str, Any]]:
    """Open-market purchases (non-derivative, code P, acquired) from one
    Form 4 ownershipDocument. Filings under a 10b5-1 plan are excluded
    entirely; exercises, grants, withholding, gifts, and sales never match
    the code filter. Malformed XML yields no rows."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    if _flag(root, "aff10b5One"):
        return []
    symbol_node = root.find("issuer/issuerTradingSymbol")
    symbol = (symbol_node.text or "").strip() if symbol_node is not None else ""
    owner_node = root.find("reportingOwner/reportingOwnerId/rptOwnerName")
    owner = (owner_node.text or "").strip() if owner_node is not None else ""
    relationship = "reportingOwner/reportingOwnerRelationship"
    title_node = root.find(f"{relationship}/officerTitle")
    rows: list[dict[str, Any]] = []
    for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code_node = txn.find("transactionCoding/transactionCode")
        code = (code_node.text or "").strip() if code_node is not None else ""
        acquired = _element_value(txn.find("transactionAmounts/transactionAcquiredDisposedCode"))
        if code != "P" or acquired != "A":
            continue
        shares = _element_value(txn.find("transactionAmounts/transactionShares"))
        price = _element_value(txn.find("transactionAmounts/transactionPricePerShare"))
        rows.append(
            {
                "symbol": symbol,
                "owner_name": owner,
                "is_director": _flag(root, f"{relationship}/isDirector"),
                "is_officer": _flag(root, f"{relationship}/isOfficer"),
                "officer_title": (
                    (title_node.text or "").strip() if title_node is not None else ""
                ),
                "is_ten_percent_owner": _flag(root, f"{relationship}/isTenPercentOwner"),
                "transaction_date": _element_value(txn.find("transactionDate")),
                "shares": float(shares) if shares else None,
                "price_per_share": float(price) if price else None,
            }
        )
    return rows


def _pad_cik(cik: int | str) -> str:
    return str(cik).zfill(10)


def parse_company_browse_atom(
    atom_text: str,
    *,
    queried_ticker: str,
    source_url: str,
) -> EdgarTickerDiscoveryCandidate | None:
    """Parse one SEC company-browse result without assigning PIT semantics."""
    try:
        root = ET.fromstring(atom_text)
    except ET.ParseError as exc:
        raise ProviderError("SEC company-browse response is malformed XML") from exc

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ProviderError("SEC company-browse response is not an Atom feed")
    company = root.find("atom:company-info", namespace)
    if company is None:
        return None
    cik = company.findtext("atom:cik", namespaces=namespace)
    title = company.findtext("atom:conformed-name", namespaces=namespace)
    if not cik or not cik.isdigit() or not title:
        raise ProviderError("SEC company-browse identity is incomplete or invalid")
    return EdgarTickerDiscoveryCandidate(
        cik=_pad_cik(cik),
        queried_ticker=queried_ticker,
        title=title.strip(),
        source_url=source_url,
    )


def _parse_acceptance(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)


def _extract_2_02(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract 8-K rows carrying Item 2.02 from EDGAR's parallel-array shape."""
    return [
        {
            "filing_date": row["filing_date"],
            "acceptance_datetime": row["acceptance_datetime"],
            "accession_number": row["accession_number"],
            "items": row["items"],
        }
        for row in _extract_8k(doc, {"2.02"})
    ]


def _parallel_value(doc: dict[str, Any], field: str, index: int) -> Any:
    values = doc.get(field, [])
    return values[index] if index < len(values) else None


def _extract_8k(
    doc: dict[str, Any],
    item_types: Collection[str],
) -> list[dict[str, Any]]:
    """Extract exact-token 8-K item matches from EDGAR's parallel arrays."""
    forms = doc.get("form", [])
    wanted = frozenset(item_types)
    rows: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = (_parallel_value(doc, "items", i) or "").strip()
        matched = [
            token for token in (part.strip() for part in items.split(",")) if token in wanted
        ]
        if not matched:
            continue
        rows.append(
            {
                "filing_date": date.fromisoformat(doc["filingDate"][i]),
                "acceptance_datetime": _parse_acceptance(doc["acceptanceDateTime"][i]),
                "accession_number": doc["accessionNumber"][i],
                "items": items,
                "matched_items": matched,
                "primary_document": _parallel_value(doc, "primaryDocument", i),
                "primary_document_description": _parallel_value(doc, "primaryDocDescription", i),
            }
        )
    return rows


class SECEdgarProvider:
    """Filing-event provider over the free SEC EDGAR JSON endpoints."""

    name = "sec_edgar"

    def __init__(
        self,
        user_agent: str,
        *,
        timeout: float = 60.0,
        min_interval_seconds: float = 0.125,
        max_request_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if not user_agent:
            raise ConfigurationError("SEC EDGAR requires a contact user_agent (fair-use policy)")
        if max_request_attempts < 1:
            raise ValueError("max_request_attempts must be at least 1")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self._user_agent = user_agent
        self._timeout = timeout
        self._max_request_attempts = max_request_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self.rate_limiter = RateLimiter(min_interval_seconds=min_interval_seconds)
        self._client: httpx.Client | None = None
        self._ticker_entries: tuple[EdgarTickerCandidate, ...] | None = None

    # -- transport ---------------------------------------------------------
    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "User-Agent": self._user_agent,
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip,deflate",
                },
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    def _get_response(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """GET with bounded retry for transport failures and server errors."""
        for attempt in range(self._max_request_attempts):
            self.rate_limiter.acquire()
            try:
                response = self._get_client().get(url, headers=headers)
            except httpx.RequestError as exc:
                if attempt + 1 == self._max_request_attempts:
                    raise ProviderError(f"SEC EDGAR request failed: {exc}") from exc
            else:
                if response.status_code == 429:
                    raise RateLimitError("SEC EDGAR rate limit exceeded")
                if 500 <= response.status_code < 600:
                    if attempt + 1 == self._max_request_attempts:
                        raise ProviderError(
                            f"SEC EDGAR returned HTTP {response.status_code} for {url}"
                        )
                elif response.status_code != 200:
                    raise ProviderError(f"SEC EDGAR returned HTTP {response.status_code} for {url}")
                else:
                    return response

            delay = self._retry_backoff_seconds * (2**attempt)
            if delay > 0:
                time.sleep(delay)

        raise AssertionError("SEC EDGAR retry loop exhausted without returning or raising")

    def _get_json(self, url: str) -> Any:
        return self._get_response(url).json()

    # -- reference lookups ---------------------------------------------------
    def _ticker_reference_entries(self) -> tuple[EdgarTickerCandidate, ...]:
        if self._ticker_entries is None:
            payload = self._get_json(TICKERS_URL)
            try:
                entries = {
                    EdgarTickerCandidate(
                        cik=_pad_cik(entry["cik_str"]),
                        ticker=str(entry["ticker"]).upper(),
                        title=str(entry["title"]),
                    )
                    for entry in payload.values()
                }
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ProviderError("SEC ticker reference payload is malformed") from exc
            self._ticker_entries = tuple(
                sorted(entries, key=lambda entry: (entry.ticker, entry.cik, entry.title))
            )
        return self._ticker_entries

    def ticker_candidates(self, symbol: str) -> tuple[EdgarTickerCandidate, ...]:
        """Return every exact current SEC association for ``symbol``.

        The SEC describes this lookup as current search metadata and does not
        guarantee its accuracy or scope. Callers must not treat a candidate as
        point-in-time identity evidence. Ambiguity is preserved rather than
        silently resolving duplicate ticker associations.
        """
        normalized = symbol.strip().upper().replace("-", ".")
        return tuple(
            entry
            for entry in self._ticker_reference_entries()
            if entry.ticker.replace("-", ".") == normalized
        )

    def resolve_cik(self, symbol: str) -> str | None:
        """Return the zero-padded CIK for a ticker, or ``None`` if unknown.

        Hyphenated share classes fall back to EDGAR's dot form (BRK-B → BRK.B).
        """
        candidates = self.ticker_candidates(symbol)
        ciks = {candidate.cik for candidate in candidates}
        if len(ciks) != 1:
            return None
        return next(iter(ciks))

    def ticker_discovery_candidate(self, symbol: str) -> EdgarTickerDiscoveryCandidate | None:
        """Resolve an SEC browse candidate for a current or legacy ticker.

        The browse result is discovery metadata only. Callers must confirm the
        candidate against accession-bound filing evidence before assigning a
        point-in-time identity interval.
        """
        normalized = symbol.strip().upper().replace("-", ".")
        url = httpx.URL(
            COMPANY_BROWSE_URL,
            params={
                "action": "getcompany",
                "CIK": normalized,
                "owner": "exclude",
                "count": "10",
                "output": "atom",
            },
        )
        for attempt in range(self._max_request_attempts):
            response = self._get_response(
                str(url),
                headers={"Accept": "application/atom+xml"},
            )
            try:
                return parse_company_browse_atom(
                    response.text,
                    queried_ticker=normalized,
                    source_url=str(response.request.url),
                )
            except ProviderError:
                if attempt + 1 == self._max_request_attempts:
                    raise
                base_delay = self._retry_backoff_seconds * (2**attempt)
                delay = base_delay + random.uniform(0.0, base_delay * 0.25)  # noqa: S311
                if delay > 0:
                    time.sleep(delay)
        raise AssertionError("SEC company-browse retry loop exhausted")

    def _filing_payloads(self, cik10: str) -> list[tuple[str, dict[str, Any]]]:
        submissions_url = SUBMISSIONS_URL.format(cik10=cik10)
        submissions = self._get_json(submissions_url)
        try:
            filings = submissions["filings"]
            recent = filings["recent"]
            archives = filings.get("files", [])
        except (KeyError, TypeError) as exc:
            raise ProviderError("SEC submissions payload is malformed") from exc

        payloads = [(submissions_url, recent)]
        for archive in archives:
            try:
                archive_url = SUBMISSIONS_ARCHIVE_URL.format(name=archive["name"])
            except (KeyError, TypeError) as exc:
                raise ProviderError("SEC submissions archive reference is malformed") from exc
            payloads.append((archive_url, self._get_json(archive_url)))
        return payloads

    # -- events ---------------------------------------------------------------
    def _symbol_filing_docs(self, cik10: str) -> list[dict[str, Any]]:
        return [payload for _source_url, payload in self._filing_payloads(cik10)]

    def filing_index(
        self,
        cik: int | str,
        *,
        start: date,
        end: date,
        forms: Collection[str] | None = None,
    ) -> tuple[EdgarFilingIndexEntry, ...]:
        """Return deterministic accession references inside an inclusive window."""
        if end < start:
            raise ValueError(f"end ({end}) must be on or after start ({start})")
        cik10 = _pad_cik(cik)
        wanted = frozenset(forms) if forms is not None else None
        entries: set[EdgarFilingIndexEntry] = set()
        for source_url, payload in self._filing_payloads(cik10):
            accessions = payload.get("accessionNumber", [])
            if not isinstance(accessions, list):
                raise ProviderError("SEC submissions accession array is malformed")
            for index, accession in enumerate(accessions):
                raw_date = _parallel_value(payload, "filingDate", index)
                form = _parallel_value(payload, "form", index)
                if not accession or not raw_date or not form:
                    raise ProviderError("SEC submissions filing index contains incomplete rows")
                try:
                    filing_date = date.fromisoformat(str(raw_date))
                except ValueError as exc:
                    raise ProviderError("SEC submissions filing date is malformed") from exc
                form_text = str(form)
                if start <= filing_date <= end and (wanted is None or form_text in wanted):
                    entries.add(
                        EdgarFilingIndexEntry(
                            cik=cik10,
                            filing_date=filing_date,
                            accession_number=str(accession),
                            form=form_text,
                            source_url=source_url,
                        )
                    )
        return tuple(
            sorted(
                entries,
                key=lambda entry: (-entry.filing_date.toordinal(), entry.accession_number),
            )
        )

    def filing_clock_index(
        self,
        cik: int | str,
        *,
        start: date,
        end: date,
        forms: Collection[str] | None = None,
    ) -> tuple[EdgarFilingClockEntry, ...]:
        """Return accession references with official acceptance timestamps.

        Unlike :meth:`filing_index`, this method rejects a selected filing
        whose acceptance timestamp is absent or malformed. That fail-closed
        contract is required when filings are assigned to market sessions.
        """
        if end < start:
            raise ValueError(f"end ({end}) must be on or after start ({start})")
        cik10 = _pad_cik(cik)
        wanted = frozenset(forms) if forms is not None else None
        entries: set[EdgarFilingClockEntry] = set()
        for source_url, payload in self._filing_payloads(cik10):
            forms_array = payload.get("form", [])
            if not isinstance(forms_array, list):
                raise ProviderError("SEC submissions form array is malformed")
            for index, raw_form in enumerate(forms_array):
                form = str(raw_form)
                if wanted is not None and form not in wanted:
                    continue
                raw_date = _parallel_value(payload, "filingDate", index)
                accession = _parallel_value(payload, "accessionNumber", index)
                if not raw_date or not accession:
                    raise ProviderError("SEC submissions filing clock contains incomplete rows")
                try:
                    filing_date = date.fromisoformat(str(raw_date))
                except ValueError as exc:
                    raise ProviderError("SEC submissions filing date is malformed") from exc
                if not start <= filing_date <= end:
                    continue
                raw_acceptance = _parallel_value(payload, "acceptanceDateTime", index)
                if not raw_acceptance:
                    raise ProviderError("SEC submissions filing clock contains incomplete rows")
                try:
                    acceptance = _parse_acceptance(str(raw_acceptance))
                except ValueError as exc:
                    raise ProviderError(
                        "SEC submissions acceptance timestamp is malformed"
                    ) from exc
                entries.add(
                    EdgarFilingClockEntry(
                        cik=cik10,
                        filing_date=filing_date,
                        acceptance_datetime=acceptance,
                        accession_number=str(accession),
                        form=form,
                        source_url=source_url,
                    )
                )
        return tuple(
            sorted(
                entries,
                key=lambda entry: (-entry.filing_date.toordinal(), entry.accession_number),
            )
        )

    def fetch_earnings_events(
        self,
        symbols: list[str],
        *,
        start: date,
        end: date,
    ) -> pl.DataFrame:
        """Fetch 8-K/2.02 events for ``symbols`` with filing dates in [start, end].

        Unknown symbols are skipped (callers needing per-symbol error accounting
        use :meth:`resolve_cik` first).
        """
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            cik10 = self.resolve_cik(symbol)
            if cik10 is None:
                continue
            for doc in self._symbol_filing_docs(cik10):
                for event in _extract_2_02(doc):
                    if start <= event["filing_date"] <= end:
                        rows.append({"symbol": symbol, "cik": cik10, **event})
        return pl.DataFrame(rows, schema=_EVENT_SCHEMA)

    def fetch_8k_events(
        self,
        symbols: list[str],
        *,
        start: date,
        end: date,
        item_types: Collection[str] = DEFAULT_8K_ITEMS,
    ) -> pl.DataFrame:
        """Fetch exact-token 8-K item events with filing-clock metadata.

        The returned symbol is the caller-supplied lookup symbol. It is not a
        claim that the ticker was valid on the filing date; research consumers
        must verify CIK-to-ticker identity point in time before eligibility.
        """
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            cik10 = self.resolve_cik(symbol)
            if cik10 is None:
                continue
            rows.extend(
                self._fetch_8k_event_rows(
                    cik10,
                    lookup_symbol=symbol,
                    start=start,
                    end=end,
                    item_types=item_types,
                )
            )
        return pl.DataFrame(rows, schema=_GENERIC_8K_EVENT_SCHEMA)

    def fetch_8k_events_for_cik(
        self,
        cik: int | str,
        *,
        lookup_symbol: str,
        start: date,
        end: date,
        item_types: Collection[str] = DEFAULT_8K_ITEMS,
    ) -> pl.DataFrame:
        """Fetch 8-K events for an explicit discovery CIK.

        ``lookup_symbol`` is discovery provenance only. Accession metadata
        must confirm the filing symbol before research eligibility is assigned.
        """
        rows = self._fetch_8k_event_rows(
            _pad_cik(cik),
            lookup_symbol=lookup_symbol,
            start=start,
            end=end,
            item_types=item_types,
        )
        return pl.DataFrame(rows, schema=_GENERIC_8K_EVENT_SCHEMA)

    def _fetch_8k_event_rows(
        self,
        cik10: str,
        *,
        lookup_symbol: str,
        start: date,
        end: date,
        item_types: Collection[str],
    ) -> list[dict[str, Any]]:
        wanted = frozenset(item_types)
        if not wanted:
            raise ValueError("item_types must contain at least one SEC item code")
        rows: list[dict[str, Any]] = []
        for doc in self._symbol_filing_docs(cik10):
            for event in _extract_8k(doc, wanted):
                if start <= event["filing_date"] <= end:
                    rows.append(
                        {
                            "symbol": lookup_symbol,
                            "cik": cik10,
                            **event,
                        }
                    )
        return rows

    def fetch_accession_metadata(
        self,
        cik: int | str,
        accession_number: str,
    ) -> EdgarAccessionMetadata:
        """Fetch accession-bound filing symbols and attachment types.

        The complete-submission SGML is filing-date evidence. Missing trading
        symbols remain missing; this method never falls back to the provider's
        current ticker map.
        """
        url = complete_submission_url(cik, accession_number)
        submission_text = self._get_text(url)
        return parse_edgar_accession_metadata(
            submission_text,
            cik=cik,
            accession_number=accession_number,
        )

    def fetch_accession_text(
        self,
        cik: int | str,
        accession_number: str,
    ) -> str:
        """Fetch one official complete-submission SGML payload."""
        return self._get_text(complete_submission_url(cik, accession_number))

    def get_corporate_actions(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """Extract explicitly disclosed split and dividend evidence.

        Candidate 8-K filings are searched from a bounded pre-window lookback
        because a declared action can become effective after its filing date.
        Returned rows are included only when an explicit action date, or as a
        last resort the filing date, lies inside the inclusive requested range.
        Missing ex-dates and other facts remain ``None``; no calendar dates are
        inferred from SEC record or payment dates.
        """
        if end < start:
            raise ValueError(f"end ({end}) must be on or after start ({start})")

        from liq.data.providers.sec_edgar_actions import (
            EdgarMarkupError,
            action_range_match_basis,
            extract_pdf_text,
            parse_edgar_corporate_actions,
        )

        filing_start = start - timedelta(days=CORPORATE_ACTION_FILING_LOOKBACK_DAYS)
        events = self.fetch_8k_events(
            [symbol],
            start=filing_start,
            end=end,
            item_types=CORPORATE_ACTION_8K_ITEMS,
        )
        actions: list[dict[str, Any]] = []
        for event in events.rows(named=True):
            event_cik = str(event["cik"])
            event_accession = str(event["accession_number"])
            submission_text = self.fetch_accession_text(
                event_cik,
                event_accession,
            )

            def resolve_pdf_text(
                document: EdgarFilingDocument,
                *,
                cik: str = event_cik,
                accession: str = event_accession,
            ) -> str:
                if document.filename is None:
                    raise EdgarMarkupError("SEC PDF attachment is missing its filename")
                url = sec_archive_document_url(
                    cik,
                    accession,
                    document.filename,
                )
                return extract_pdf_text(self._get_response(url).content)

            try:
                parsed = parse_edgar_corporate_actions(
                    submission_text,
                    symbol=str(event["symbol"]),
                    cik=str(event["cik"]),
                    filing_date=event["filing_date"],
                    acceptance_datetime=event["acceptance_datetime"],
                    accession_number=event_accession,
                    pdf_text_resolver=resolve_pdf_text,
                )
            except EdgarMarkupError as exc:
                raise ProviderError(
                    "SEC EDGAR corporate-action parse failed for accession "
                    f"{event_accession}: {exc}"
                ) from exc
            for action in parsed:
                basis = action_range_match_basis(action, start=start, end=end)
                if basis is not None:
                    actions.append({**action, "range_match_basis": basis})
        return actions

    def _get_text(self, url: str) -> str:
        return self._get_response(url).text

    def fetch_form4_purchases(
        self,
        symbol: str,
        *,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """Open-market Form 4 purchases for ``symbol`` filed in [start, end].

        One raw ownershipDocument fetch per Form 4 filing in the window;
        each purchase row carries the filing's acceptance datetime and
        accession for entry-delay rules and provenance. Unknown symbols
        return no rows.
        """
        cik10 = self.resolve_cik(symbol)
        if cik10 is None:
            return []
        purchases: list[dict[str, Any]] = []
        for doc in self._symbol_filing_docs(cik10):
            recent = doc.get("filings", {}).get("recent", doc)
            forms = recent.get("form", [])
            for index, form in enumerate(forms):
                if form != "4":
                    continue
                filing_date = date.fromisoformat(recent["filingDate"][index])
                if not (start <= filing_date <= end):
                    continue
                url = raw_form4_url(
                    int(cik10),
                    recent["accessionNumber"][index],
                    recent["primaryDocument"][index],
                )
                try:
                    xml_text = self._get_text(url)
                except httpx.HTTPStatusError:
                    continue  # missing/renamed document: skip, counted by caller diff
                for row in parse_form4_purchases(xml_text):
                    purchases.append(
                        {
                            **row,
                            "filing_date": str(filing_date),
                            "acceptance_datetime": recent["acceptanceDateTime"][index],
                            "accession": recent["accessionNumber"][index],
                            "list_symbol": symbol,
                        }
                    )
        return purchases

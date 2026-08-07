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

import re
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from typing import Any

import httpx
import polars as pl

from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.rate_limiter import RateLimiter

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SUBMISSIONS_ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"
COMPLETE_SUBMISSION_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}.txt"
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


def _sgml_field(document: str, field: str) -> str | None:
    match = re.search(rf"(?im)^\s*<{field}>\s*([^\r\n<]+)", document)
    return match.group(1).strip() if match else None


def _sgml_text(document: str) -> str:
    match = re.search(r"<TEXT>\s*(.*?)</TEXT>", document, re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def _filing_symbols(primary_documents: list[str]) -> tuple[str, ...]:
    symbols: list[str] = []
    seen: set[str] = set()
    for document in primary_documents:
        parser = _TradingSymbolParser()
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
    for document in _DOCUMENT_PATTERN.findall(submission_text):
        document_type = _sgml_field(document, "TYPE")
        if document_type is None:
            continue
        normalized_type = document_type.upper()
        document_types.append(normalized_type)
        if normalized_type == "8-K":
            primary_documents.append(_sgml_text(document))
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
        timeout: float = 30.0,
        min_interval_seconds: float = 0.125,
    ) -> None:
        if not user_agent:
            raise ConfigurationError("SEC EDGAR requires a contact user_agent (fair-use policy)")
        self._user_agent = user_agent
        self._timeout = timeout
        self.rate_limiter = RateLimiter(min_interval_seconds=min_interval_seconds)
        self._client: httpx.Client | None = None
        self._ticker_map: dict[str, str] | None = None

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

    def _get_json(self, url: str) -> Any:
        self.rate_limiter.acquire()
        try:
            response = self._get_client().get(url)
        except httpx.RequestError as exc:
            raise ProviderError(f"SEC EDGAR request failed: {exc}") from exc
        if response.status_code == 429:
            raise RateLimitError("SEC EDGAR rate limit exceeded")
        if response.status_code != 200:
            raise ProviderError(f"SEC EDGAR returned HTTP {response.status_code} for {url}")
        return response.json()

    # -- reference lookups ---------------------------------------------------
    def _tickers(self) -> dict[str, str]:
        if self._ticker_map is None:
            payload = self._get_json(TICKERS_URL)
            self._ticker_map = {
                entry["ticker"].upper(): _pad_cik(entry["cik_str"]) for entry in payload.values()
            }
        return self._ticker_map

    def resolve_cik(self, symbol: str) -> str | None:
        """Return the zero-padded CIK for a ticker, or ``None`` if unknown.

        Hyphenated share classes fall back to EDGAR's dot form (BRK-B → BRK.B).
        """
        tickers = self._tickers()
        upper = symbol.upper()
        return tickers.get(upper) or tickers.get(upper.replace("-", "."))

    # -- events ---------------------------------------------------------------
    def _symbol_filing_docs(self, cik10: str) -> list[dict[str, Any]]:
        submissions = self._get_json(SUBMISSIONS_URL.format(cik10=cik10))
        filings = submissions.get("filings", {})
        docs = [filings.get("recent", {})]
        for archive in filings.get("files", []):
            docs.append(self._get_json(SUBMISSIONS_ARCHIVE_URL.format(name=archive["name"])))
        return docs

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
        wanted = frozenset(item_types)
        if not wanted:
            raise ValueError("item_types must contain at least one SEC item code")
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            cik10 = self.resolve_cik(symbol)
            if cik10 is None:
                continue
            for doc in self._symbol_filing_docs(cik10):
                for event in _extract_8k(doc, wanted):
                    if start <= event["filing_date"] <= end:
                        rows.append({"symbol": symbol, "cik": cik10, **event})
        return pl.DataFrame(rows, schema=_GENERIC_8K_EVENT_SCHEMA)

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

    def _get_text(self, url: str) -> str:
        self.rate_limiter.acquire()
        response = self._get_client().get(url)
        response.raise_for_status()
        return response.text

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

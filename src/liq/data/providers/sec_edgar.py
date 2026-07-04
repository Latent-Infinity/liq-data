"""SEC EDGAR filings provider (reference/event data, not OHLCV).

Fetches 8-K Item 2.02 (earnings press release) filing events per symbol from
the free SEC EDGAR endpoints, using ``acceptanceDateTime`` as the tradable
announcement-time proxy.

Deliberately NOT a :class:`~liq.data.providers.base.BaseProvider` subclass:
that ABC's contract is OHLCV bars (``fetch_bars``/``list_instruments``), which
an event provider cannot honestly implement (interface segregation). This is
the stack's first event/reference adapter and defines its own narrow surface.

SEC fair-use etiquette: a real contact ``User-Agent`` is required (no default),
and requests are throttled to 8/sec via the shared :class:`RateLimiter`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx
import polars as pl

from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.rate_limiter import RateLimiter

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SUBMISSIONS_ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"

_EVENT_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "symbol": pl.String,
    "cik": pl.String,
    "filing_date": pl.Date,
    "acceptance_datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    "accession_number": pl.String,
    "items": pl.String,
}


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
    forms = doc.get("form", [])
    rows: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items_raw = doc.get("items", [])
        items = (items_raw[i] if i < len(items_raw) else "") or ""
        if "2.02" not in [token.strip() for token in items.split(",")]:
            continue
        rows.append(
            {
                "filing_date": date.fromisoformat(doc["filingDate"][i]),
                "acceptance_datetime": _parse_acceptance(doc["acceptanceDateTime"][i]),
                "accession_number": doc["accessionNumber"][i],
                "items": items,
            }
        )
    return rows


class SECEdgarProvider:
    """Earnings-event provider over the free SEC EDGAR JSON endpoints."""

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

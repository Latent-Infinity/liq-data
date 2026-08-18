"""Corporate-action evidence extracted from real SEC EDGAR filings.

The reduced SGML fixtures retain only the action-bearing language and filing
envelope from the named accessions. They are real public filing text, not
synthetic financial data. Transport tests stay mocked; live SEC responses are
readiness evidence, not unit-test dependencies.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, date, datetime
from types import SimpleNamespace

import httpx
import pytest
import respx

from liq.data.exceptions import ProviderError
from liq.data.providers.sec_edgar import EdgarFilingDocument, SECEdgarProvider
from liq.data.providers.sec_edgar_actions import (
    EdgarMarkupError,
    extract_pdf_text,
    parse_edgar_corporate_actions,
)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
AAPL_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
AAPL_ACCESSION_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/000032019320000060/0000320193-20-000060.txt"
)

TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
}

# SEC accession 0000320193-20-000060, filed 2020-07-30. Reduced to the
# action-bearing EX-99.1 paragraph fetched from the official archive.
AAPL_2020_SUBMISSION = """\
<SEC-DOCUMENT>0000320193-20-000060.txt
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>aapl-20200730.htm
<DESCRIPTION>8-K
<TEXT><html><body>Item 2.02. Results of Operations and Financial Condition.</body></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>aapl-20200730xex991.htm
<DESCRIPTION>PRESS RELEASE
<TEXT><html><body>
Apple's Board of Directors has declared a cash dividend of $0.82 per share of
the Company's common stock. The dividend is payable on August 13, 2020 to
shareholders of record as of the close of business on August 10, 2020.
The Board of Directors has also approved a four-for-one stock split to make the
stock more accessible to a broader base of investors. Each Apple shareholder
of record at the close of business on August 24, 2020 will receive three
additional shares for every share held on the record date, and trading will
begin on a split-adjusted basis on August 31, 2020.
Plans include the four-for-one stock split described above.
</body></html></TEXT>
</DOCUMENT>
"""

# SEC accession 0001564590-20-039353 for Tesla CIK 0001318605, filed
# 2020-08-11. The phrase "stock dividend" must remain a split, not a cash
# dividend.
TSLA_2020_SUBMISSION = """\
<SEC-DOCUMENT>0001564590-20-039353.txt
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>tsla-ex991_6.htm
<DESCRIPTION>EX-99.1
<TEXT><html><body>
Tesla Announces a Five-for-One Stock Split. Tesla announced today that the
Board of Directors has approved and declared a five-for-one split of Tesla's
common stock in the form of a stock dividend. Each stockholder of record on
August 21, 2020 will receive a dividend of four additional shares of common
stock for each then-held share, to be distributed after close of trading on
August 28, 2020. Trading will begin on a stock split-adjusted basis on
August 31, 2020.
</body></html></TEXT>
</DOCUMENT>
"""

# SEC accession 0001045810-24-000113, filed 2024-05-22. Reduced from the
# official 8-K Item 8.01 text.
NVDA_2024_SUBMISSION = """\
<SEC-DOCUMENT>0001045810-24-000113.txt
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>nvda-20240522.htm
<DESCRIPTION>8-K
<TEXT><html><body>
On May 22, 2024, the Company announced a ten-for-one forward stock split.
As a result of the Stock Split, each record holder of common stock as of the
close of market on Thursday, June 6, 2024 will receive nine additional shares
of common stock, to be distributed after the close of market on Friday,
June 7, 2024. Trading is expected to commence on a split-adjusted basis at
market open on Monday, June 10, 2024. The Company also increased its quarterly
cash dividend by 150 percent from $0.04 per share to $0.10 per share of common
stock. The increased dividend will be paid on Friday, June 28, 2024, to all
shareholders of record on Tuesday, June 11, 2024.
</body></html></TEXT>
</DOCUMENT>
"""


def _parse(
    text: str,
    *,
    symbol: str,
    cik: str,
    accession: str,
    filed: date,
) -> list[dict[str, object]]:
    return parse_edgar_corporate_actions(
        text,
        symbol=symbol,
        cik=cik,
        filing_date=filed,
        acceptance_datetime=datetime.combine(filed, datetime.min.time(), tzinfo=UTC),
        accession_number=accession,
    )


class TestPureParser:
    @pytest.mark.parametrize(
        ("document_type", "filename", "payload"),
        [
            (
                "ZIP",
                "0001193125-20-039203-xbrl.zip",
                "<![IR binary-archive-bytes]]>",
            ),
            (
                "GRAPHIC",
                "earningspresentation014.jpg",
                "<![F binary-image-bytes]]>",
            ),
            (
                "EXCEL",
                "Financial_Report.xlsx",
                "PK binary-spreadsheet-bytes",
            ),
        ],
    )
    def test_skips_verified_non_text_packaging_attachment(
        self,
        document_type: str,
        filename: str,
        payload: str,
    ) -> None:
        non_text_document = f"""\
<DOCUMENT>
<TYPE>{document_type}
<FILENAME>{filename}
<TEXT>{payload}</TEXT>
</DOCUMENT>
"""

        actions = _parse(
            non_text_document + AAPL_2020_SUBMISSION,
            symbol="AAPL",
            cik="0000320193",
            accession="0000320193-20-000060",
            filed=date(2020, 7, 30),
        )

        assert [row["corporate_action_type"] for row in actions] == [
            "cash_dividend",
            "stock_split",
        ]

    def test_pdf_exhibit_with_unsupported_markup_remains_fail_closed(self) -> None:
        submission = """\
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>potential-action-disclosure.pdf
<TEXT><![F binary-pdf-bytes]]></TEXT>
</DOCUMENT>
"""

        with pytest.raises(
            EdgarMarkupError,
            match=r"document type EX-99\.1.*potential-action-disclosure\.pdf.*<!\[F",
        ):
            _parse(
                submission,
                symbol="AAPL",
                cik="0000320193",
                accession="0000320193-20-000060",
                filed=date(2020, 7, 30),
            )

    def test_pdf_exhibit_uses_required_text_resolver(self) -> None:
        submission = """\
<DOCUMENT>
<TYPE>EX-99.2
<FILENAME>amdq4andfy2019earningsslides.pdf
<TEXT><![GCALEC binary-pdf-bytes]]></TEXT>
</DOCUMENT>
"""
        resolved_documents: list[tuple[str, str | None]] = []

        def resolve_pdf(document: EdgarFilingDocument) -> str:
            resolved_documents.append((document.document_type, document.filename))
            return "FOURTH QUARTER AND FULL YEAR 2019 FINANCIAL RESULTS JANUARY 28, 2020"

        actions = parse_edgar_corporate_actions(
            submission,
            symbol="AMD",
            cik="0000002488",
            filing_date=date(2020, 2, 4),
            acceptance_datetime=datetime(2020, 2, 4, tzinfo=UTC),
            accession_number="0000002488-20-000006",
            pdf_text_resolver=resolve_pdf,
        )

        assert actions == []
        assert resolved_documents == [("EX-99.2", "amdq4andfy2019earningsslides.pdf")]

    def test_pdf_text_extractor_validates_magic(self) -> None:
        with pytest.raises(EdgarMarkupError, match="does not start with PDF magic"):
            extract_pdf_text(b"not a pdf")

    def test_pdf_text_extractor_fails_closed_when_executable_is_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("liq.data.providers.sec_edgar_actions.shutil.which", lambda _: None)

        with pytest.raises(EdgarMarkupError, match="pdftotext executable is unavailable"):
            extract_pdf_text(b"%PDF-real-public-filing-bytes")

    def test_pdf_text_extractor_returns_normalized_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "liq.data.providers.sec_edgar_actions.shutil.which",
            lambda _: "/usr/bin/pdftotext",
        )
        observed: dict[str, object] = {}

        def run(command: list[str], **kwargs: object) -> object:
            observed["command"] = command
            observed.update(kwargs)
            return SimpleNamespace(returncode=0, stdout=b"AMD  quarterly\nresults\n", stderr=b"")

        monkeypatch.setattr("liq.data.providers.sec_edgar_actions.subprocess.run", run)

        text = extract_pdf_text(b"%PDF-real-public-filing-bytes")

        assert text == "AMD quarterly results"
        assert observed["command"] == ["/usr/bin/pdftotext", "-layout", "-", "-"]
        assert observed["input"] == b"%PDF-real-public-filing-bytes"
        assert observed["timeout"] == 30.0

    @pytest.mark.parametrize(
        ("failure", "message"),
        [
            (
                subprocess.TimeoutExpired(cmd="pdftotext", timeout=30.0),
                "timed out",
            ),
            (SimpleNamespace(returncode=1, stdout=b"", stderr=b"bad pdf"), "exited with code 1"),
            (SimpleNamespace(returncode=0, stdout=b"  \n", stderr=b""), "no extractable text"),
        ],
    )
    def test_pdf_text_extractor_fails_closed_on_extraction_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure: object,
        message: str,
    ) -> None:
        monkeypatch.setattr(
            "liq.data.providers.sec_edgar_actions.shutil.which",
            lambda _: "/usr/bin/pdftotext",
        )

        def run(*args: object, **kwargs: object) -> object:
            del args, kwargs
            if isinstance(failure, BaseException):
                raise failure
            return failure

        monkeypatch.setattr("liq.data.providers.sec_edgar_actions.subprocess.run", run)

        with pytest.raises(EdgarMarkupError, match=message):
            extract_pdf_text(b"%PDF-real-public-filing-bytes")

    def test_extracts_aapl_split_and_cash_dividend_without_inventing_ex_date(self) -> None:
        actions = _parse(
            AAPL_2020_SUBMISSION,
            symbol="AAPL",
            cik="0000320193",
            accession="0000320193-20-000060",
            filed=date(2020, 7, 30),
        )

        assert [row["corporate_action_type"] for row in actions] == [
            "cash_dividend",
            "stock_split",
        ]
        dividend, split = actions
        assert dividend["cash_amount"] == "0.82"
        assert dividend["record_date"] == "2020-08-10"
        assert dividend["payment_date"] == "2020-08-13"
        assert dividend["ex_date"] is None
        assert dividend["parse_status"] == "MISSING_EX_DATE"

        assert split["split_ratio_from"] == 1
        assert split["split_ratio_to"] == 4
        assert split["record_date"] == "2020-08-24"
        assert split["effective_date"] == "2020-08-31"
        assert split["parse_status"] == "EXPLICIT_EFFECTIVE_DATE_AND_RATIO"
        assert split["source_document_type"] == "EX-99.1"
        assert split["source_accession"] == "0000320193-20-000060"
        assert len(str(split["evidence_sha256"])) == 64

    def test_stock_dividend_wording_is_a_split_not_a_cash_dividend(self) -> None:
        actions = _parse(
            TSLA_2020_SUBMISSION,
            symbol="TSLA",
            cik="0001318605",
            accession="0001564590-20-039353",
            filed=date(2020, 8, 11),
        )

        assert len(actions) == 1
        split = actions[0]
        assert split["corporate_action_type"] == "stock_split"
        assert split["split_ratio_from"] == 1
        assert split["split_ratio_to"] == 5
        assert split["record_date"] == "2020-08-21"
        assert split["distribution_date"] == "2020-08-28"
        assert split["effective_date"] == "2020-08-31"

    def test_handles_weekday_dates_and_changed_dividend_amount(self) -> None:
        actions = _parse(
            NVDA_2024_SUBMISSION,
            symbol="NVDA",
            cik="0001045810",
            accession="0001045810-24-000113",
            filed=date(2024, 5, 22),
        )

        dividend, split = actions
        assert dividend["cash_amount"] == "0.10"
        assert dividend["record_date"] == "2024-06-11"
        assert dividend["payment_date"] == "2024-06-28"
        assert dividend["ex_date"] is None
        assert split["split_ratio_from"] == 1
        assert split["split_ratio_to"] == 10
        assert split["record_date"] == "2024-06-06"
        assert split["distribution_date"] == "2024-06-07"
        assert split["effective_date"] == "2024-06-10"

    def test_preserves_same_amount_dividends_with_distinct_dates(self) -> None:
        submission = """\
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>first-dividend.htm
<TEXT><html><body>
The Board declared a quarterly cash dividend of $0.10 per share, payable on
March 15, 2024 to shareholders of record on March 1, 2024.
</body></html></TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.2
<FILENAME>second-dividend.htm
<TEXT><html><body>
The Board declared a quarterly cash dividend of $0.10 per share, payable on
June 14, 2024 to shareholders of record on May 31, 2024.
</body></html></TEXT>
</DOCUMENT>
"""

        actions = _parse(
            submission,
            symbol="AAPL",
            cik="0000320193",
            accession="0000320193-24-000001",
            filed=date(2024, 1, 2),
        )

        assert [(row["record_date"], row["payment_date"]) for row in actions] == [
            ("2024-03-01", "2024-03-15"),
            ("2024-05-31", "2024-06-14"),
        ]

    def test_ignores_filing_without_explicit_action_language(self) -> None:
        assert (
            _parse(
                "<DOCUMENT><TYPE>8-K<TEXT><html><body>No action.</body></html></TEXT></DOCUMENT>",
                symbol="AAPL",
                cik="0000320193",
                accession="0000320193-20-000060",
                filed=date(2020, 7, 30),
            )
            == []
        )

    def test_unsupported_marked_section_fails_closed_with_bounded_context(self) -> None:
        submission = """\
<DOCUMENT><TYPE>8-K<TEXT><html><body>
<![UNSUPPORTED proprietary-control-data]]>
No action.
</body></html></TEXT></DOCUMENT>
"""

        with pytest.raises(
            EdgarMarkupError,
            match=r"document type 8-K.*<!\[UNSUPPORTED",
        ) as exc_info:
            _parse(
                submission,
                symbol="AAPL",
                cik="0000320193",
                accession="0000320193-20-000060",
                filed=date(2020, 7, 30),
            )

        assert "proprietary-control-data" not in str(exc_info.value)


class TestProviderIntegration:
    @respx.mock
    def test_fetches_candidate_accession_and_filters_on_action_dates(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(AAPL_SUBMISSIONS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {
                            "form": ["8-K"],
                            "filingDate": ["2020-07-30"],
                            "acceptanceDateTime": ["2020-07-30T16:31:00.000Z"],
                            "accessionNumber": ["0000320193-20-000060"],
                            "items": ["2.02,9.01"],
                            "primaryDocument": ["aapl-20200730.htm"],
                            "primaryDocDescription": ["8-K"],
                        },
                        "files": [],
                    }
                },
            )
        )
        accession = respx.get(AAPL_ACCESSION_URL).mock(
            return_value=httpx.Response(200, text=AAPL_2020_SUBMISSION)
        )

        actions = SECEdgarProvider("test test@example.com").get_corporate_actions(
            "AAPL",
            date(2020, 8, 1),
            date(2020, 8, 31),
        )

        assert len(actions) == 2
        assert {row["range_match_basis"] for row in actions} == {
            "effective_date",
            "record_date",
        }
        assert accession.call_count == 1

    @respx.mock
    def test_unknown_current_ticker_is_explicitly_empty(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))

        assert (
            SECEdgarProvider("test test@example.com").get_corporate_actions(
                "OLD",
                date(2020, 1, 1),
                date(2020, 12, 31),
            )
            == []
        )

    @respx.mock
    def test_markup_failure_names_accession_without_silently_skipping_filing(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(AAPL_SUBMISSIONS_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {
                            "form": ["8-K"],
                            "filingDate": ["2020-07-30"],
                            "acceptanceDateTime": ["2020-07-30T16:31:00.000Z"],
                            "accessionNumber": ["0000320193-20-000060"],
                            "items": ["2.02,9.01"],
                            "primaryDocument": ["aapl-20200730.htm"],
                            "primaryDocDescription": ["8-K"],
                        },
                        "files": [],
                    }
                },
            )
        )
        respx.get(AAPL_ACCESSION_URL).mock(
            return_value=httpx.Response(
                200,
                text=(
                    "<DOCUMENT><TYPE>8-K<TEXT><html><body>"
                    "<![UNSUPPORTED private]]>No action.</body></html></TEXT></DOCUMENT>"
                ),
            )
        )

        with pytest.raises(ProviderError, match="0000320193-20-000060.*UNSUPPORTED"):
            SECEdgarProvider("test test@example.com").get_corporate_actions(
                "AAPL",
                date(2020, 8, 1),
                date(2020, 8, 31),
            )

    def test_rejects_inverted_range_before_transport(self) -> None:
        with pytest.raises(ValueError, match="end"):
            SECEdgarProvider("test test@example.com").get_corporate_actions(
                "AAPL",
                date(2020, 2, 1),
                date(2020, 1, 1),
            )

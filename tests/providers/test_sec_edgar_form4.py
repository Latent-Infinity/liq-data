"""Form 4 open-market purchase extraction.

The parser is pinned on a real-schema ownershipDocument (structure verbatim
from a live filing, values reduced): nested ``<value>`` elements, boolean
role flags, the ``aff10b5One`` plan flag, and the XSL-prefixed primary
document path that must be stripped to reach the raw XML.
"""

from __future__ import annotations

import pytest

from liq.data.providers.sec_edgar import parse_form4_purchases, raw_form4_url

_FORM4_XML = """<?xml version="1.0"?>
<ownershipDocument>
  <periodOfReport>2026-06-15</periodOfReport>
  <issuer><issuerCik>0000320193</issuerCik><issuerName>Example Inc</issuerName>
    <issuerTradingSymbol>EXMP</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>true</isOfficer><officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <aff10b5One>false</aff10b5One>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-13</value></transactionDate>
      <transactionCoding><transactionFormType>4</transactionFormType>
        <transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>45.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-13</value></transactionDate>
      <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>30104</value></transactionShares>
        <transactionPricePerShare><footnoteId id="F1"/></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-06-14</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>46.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

_FORM4_10B5 = _FORM4_XML.replace("<aff10b5One>false</aff10b5One>", "<aff10b5One>true</aff10b5One>")


class TestParseForm4:
    def test_extracts_only_open_market_purchases(self) -> None:
        rows = parse_form4_purchases(_FORM4_XML)
        assert len(rows) == 1  # M (exercise) and S (sale) excluded
        row = rows[0]
        assert row["symbol"] == "EXMP"
        assert row["owner_name"] == "Doe Jane"
        assert row["is_officer"] is True
        assert row["officer_title"] == "CEO"
        assert row["is_director"] is False
        assert row["is_ten_percent_owner"] is False
        assert row["transaction_date"] == "2026-06-13"
        assert row["shares"] == pytest.approx(1000.0)
        assert row["price_per_share"] == pytest.approx(45.50)

    def test_10b5_1_plan_filings_excluded_entirely(self) -> None:
        assert parse_form4_purchases(_FORM4_10B5) == []

    def test_malformed_xml_returns_empty(self) -> None:
        assert parse_form4_purchases("<not-xml") == []


class TestRawUrl:
    def test_strips_xsl_prefix(self) -> None:
        url = raw_form4_url(320193, "0001140361-26-025622", "xslF345X06/form4.xml")
        assert url == (
            "https://www.sec.gov/Archives/edgar/data/320193/000114036126025622/form4.xml"
        )

    def test_plain_document_untouched(self) -> None:
        url = raw_form4_url(320193, "0001140361-26-025622", "form4.xml")
        assert url.endswith("/000114036126025622/form4.xml")


class TestFetchForm4Purchases:
    def test_end_to_end_with_mocked_edgar(self) -> None:
        from datetime import date

        import httpx
        import respx as respx_lib

        from liq.data.providers.sec_edgar import (
            SUBMISSIONS_URL,
            TICKERS_URL,
            SECEdgarProvider,
        )

        with respx_lib.mock:
            respx_lib.get(TICKERS_URL).mock(
                return_value=httpx.Response(200, json={"0": {"ticker": "EXMP", "cik_str": 320193}})
            )
            respx_lib.get(SUBMISSIONS_URL.format(cik10="0000320193")).mock(
                return_value=httpx.Response(
                    200,
                    json={
                        "filings": {
                            "recent": {
                                "form": ["4", "8-K"],
                                "filingDate": ["2026-06-16", "2026-06-01"],
                                "accessionNumber": [
                                    "0001140361-26-025622",
                                    "0001140361-26-020000",
                                ],
                                "primaryDocument": ["xslF345X06/form4.xml", "ex.htm"],
                                "acceptanceDateTime": [
                                    "2026-06-16T18:31:02.000Z",
                                    "2026-06-01T12:00:00.000Z",
                                ],
                                "items": ["", "2.02"],
                                "reportDate": ["2026-06-15", "2026-05-31"],
                            },
                            "files": [],
                        }
                    },
                )
            )
            respx_lib.get(
                "https://www.sec.gov/Archives/edgar/data/320193/000114036126025622/form4.xml"
            ).mock(return_value=httpx.Response(200, text=_FORM4_XML))
            provider = SECEdgarProvider(user_agent="liq-test test@example.com")
            rows = provider.fetch_form4_purchases(
                "EXMP", start=date(2026, 6, 1), end=date(2026, 6, 30)
            )
        assert len(rows) == 1
        assert rows[0]["acceptance_datetime"] == "2026-06-16T18:31:02.000Z"
        assert rows[0]["shares"] == 1000.0

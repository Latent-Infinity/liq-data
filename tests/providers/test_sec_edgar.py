"""TDD pins for the SEC EDGAR filings provider.

Response payloads mirror the real EDGAR JSON contracts (company_tickers.json
and the submissions parallel-array shape) — filing metadata, not market data.
Transport is mocked with respx following the OANDA provider test pattern.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import polars as pl
import pytest
import respx

from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.providers.sec_edgar import (
    EdgarFilingIndexEntry,
    EdgarTickerCandidate,
    SECEdgarProvider,
    complete_submission_url,
    parse_edgar_accession_metadata,
    sec_archive_document_url,
)
from liq.data.settings import LiqDataSettings, create_sec_edgar_provider

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
ARCHIVE_URL = "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"
COMPLETE_SUBMISSION_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/000032019323000075/0000320193-23-000075.txt"
)

# Reduced structural fixture from Apple accession 0000320193-23-000075. It
# retains the SEC PDS document envelope and Inline XBRL identity fact without
# copying filing narrative or financial content.
ACCESSION_SUBMISSION = """\
<SEC-DOCUMENT>0000320193-23-000075.txt
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>aapl-20230803.htm
<DESCRIPTION>8-K
<TEXT>
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL">
<body>
<ix:nonNumeric name="dei:TradingSymbol" contextRef="c-1">AAPL</ix:nonNumeric>
<ix:nonNumeric name="dei:TradingSymbol" contextRef="c-2">&#8212;</ix:nonNumeric>
</body>
</html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>a8-kex991q3202307012023.htm
<DESCRIPTION>EX-99.1
<TEXT><html><body>Attachment omitted.</body></html></TEXT>
</DOCUMENT>
"""

TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "Berkshire Hathaway"},
}

# EDGAR "filings.recent" parallel-array shape.
SUBMISSIONS_JSON = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "8-K"],
            "filingDate": ["2023-08-03", "2023-08-04", "2023-05-04"],
            "acceptanceDateTime": [
                "2023-08-03T16:30:15.000Z",
                "2023-08-04T10:00:00.000Z",
                "2023-05-04T16:31:00.000Z",
            ],
            "accessionNumber": [
                "0000320193-23-000077",
                "0000320193-23-000078",
                "0000320193-23-000064",
            ],
            "items": ["2.02,9.01", "", "2.02,9.01"],
        },
        "files": [],
    }
}


def _provider(**kwargs) -> SECEdgarProvider:
    kwargs.setdefault("retry_backoff_seconds", 0.0)
    return SECEdgarProvider(user_agent="test test@example.com", **kwargs)


class TestConstruction:
    def test_requires_user_agent(self) -> None:
        with pytest.raises(ConfigurationError, match="user_agent"):
            SECEdgarProvider(user_agent="")

    def test_name(self) -> None:
        assert _provider().name == "sec_edgar"


class TestTickerToCik:
    @respx.mock
    def test_candidate_lookup_preserves_ambiguous_current_associations(self) -> None:
        payload = {
            "0": {"cik_str": 1, "ticker": "DUP", "title": "First Corp"},
            "1": {"cik_str": 2, "ticker": "DUP", "title": "Second Corp"},
        }
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=payload))
        provider = _provider()

        assert provider.ticker_candidates("dup") == (
            EdgarTickerCandidate(cik="0000000001", ticker="DUP", title="First Corp"),
            EdgarTickerCandidate(cik="0000000002", ticker="DUP", title="Second Corp"),
        )
        assert provider.resolve_cik("DUP") is None

    @respx.mock
    def test_maps_ticker_to_padded_cik(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        provider = _provider()
        assert provider.resolve_cik("AAPL") == "0000320193"

    @respx.mock
    def test_hyphen_symbol_falls_back_to_dot_form(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        assert _provider().ticker_candidates("BRK-B") == (
            EdgarTickerCandidate(
                cik="0001067983",
                ticker="BRK.B",
                title="Berkshire Hathaway",
            ),
        )

    @respx.mock
    def test_hyphen_symbol_resolves_unique_candidate(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        assert _provider().resolve_cik("BRK-B") == "0001067983"

    @respx.mock
    def test_unknown_symbol_returns_none(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        assert _provider().resolve_cik("ZZZZ") is None


class TestFetchEarningsEvents:
    @respx.mock
    def test_extracts_8k_202_events_in_window(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=SUBMISSIONS_JSON))
        df = _provider().fetch_earnings_events(
            ["AAPL"], start=date(2023, 1, 1), end=date(2023, 12, 31)
        )
        # only the two 8-K rows carrying item 2.02; the 10-Q is excluded
        assert df.height == 2
        assert set(df.columns) == {
            "symbol",
            "cik",
            "filing_date",
            "acceptance_datetime",
            "accession_number",
            "items",
        }
        assert df["symbol"].to_list() == ["AAPL", "AAPL"]
        assert df["filing_date"].to_list() == [date(2023, 8, 3), date(2023, 5, 4)]
        first_acceptance = df["acceptance_datetime"].to_list()[0]
        assert first_acceptance == datetime(2023, 8, 3, 16, 30, 15, tzinfo=UTC)

    @respx.mock
    def test_item_match_is_exact_token_not_substring(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        payload = {
            "filings": {
                "recent": {
                    "form": ["8-K", "8-K"],
                    "filingDate": ["2023-08-03", "2023-08-04"],
                    "acceptanceDateTime": [
                        "2023-08-03T16:30:15.000Z",
                        "2023-08-04T16:30:15.000Z",
                    ],
                    "accessionNumber": ["a-1", "a-2"],
                    "items": ["5.02,9.01", "2.02"],
                },
                "files": [],
            }
        }
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=payload))
        df = _provider().fetch_earnings_events(
            ["AAPL"], start=date(2023, 1, 1), end=date(2023, 12, 31)
        )
        assert df["accession_number"].to_list() == ["a-2"]

    @respx.mock
    def test_window_filter_excludes_out_of_range(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=SUBMISSIONS_JSON))
        df = _provider().fetch_earnings_events(
            ["AAPL"], start=date(2023, 7, 1), end=date(2023, 12, 31)
        )
        assert df["filing_date"].to_list() == [date(2023, 8, 3)]

    @respx.mock
    def test_unknown_symbols_are_skipped_not_fatal(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=SUBMISSIONS_JSON))
        df = _provider().fetch_earnings_events(
            ["AAPL", "ZZZZ"], start=date(2023, 1, 1), end=date(2023, 12, 31)
        )
        assert df["symbol"].unique().to_list() == ["AAPL"]

    @respx.mock
    def test_empty_result_has_stable_schema(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=SUBMISSIONS_JSON))
        df = _provider().fetch_earnings_events(
            ["AAPL"], start=date(2010, 1, 1), end=date(2010, 12, 31)
        )
        assert df.height == 0
        assert "acceptance_datetime" in df.columns

    @respx.mock
    def test_archive_files_are_followed(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        recent_with_archive = {
            "filings": {
                "recent": SUBMISSIONS_JSON["filings"]["recent"],
                "files": [{"name": "CIK0000320193-submissions-001.json"}],
            }
        }
        archive_payload = {
            "form": ["8-K"],
            "filingDate": ["2022-02-01"],
            "acceptanceDateTime": ["2022-02-01T16:00:00.000Z"],
            "accessionNumber": ["0000320193-22-000001"],
            "items": ["2.02"],
        }
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=recent_with_archive))
        respx.get(ARCHIVE_URL).mock(return_value=httpx.Response(200, json=archive_payload))
        df = _provider().fetch_earnings_events(
            ["AAPL"], start=date(2022, 1, 1), end=date(2023, 12, 31)
        )
        assert date(2022, 2, 1) in df["filing_date"].to_list()


class TestFilingIndex:
    @respx.mock
    def test_returns_deterministic_accession_metadata_across_archives(self) -> None:
        recent_with_archive = {
            "filings": {
                "recent": SUBMISSIONS_JSON["filings"]["recent"],
                "files": [{"name": "CIK0000320193-submissions-001.json"}],
            }
        }
        archive_payload = {
            "form": ["10-K"],
            "filingDate": ["2022-02-01"],
            "accessionNumber": ["0000320193-22-000001"],
        }
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=recent_with_archive))
        respx.get(ARCHIVE_URL).mock(return_value=httpx.Response(200, json=archive_payload))

        entries = _provider().filing_index(
            "0000320193",
            start=date(2022, 1, 1),
            end=date(2023, 8, 3),
            forms={"10-K", "8-K"},
        )

        assert entries == (
            EdgarFilingIndexEntry(
                cik="0000320193",
                filing_date=date(2023, 8, 3),
                accession_number="0000320193-23-000077",
                form="8-K",
                source_url=SUBMISSIONS_URL,
            ),
            EdgarFilingIndexEntry(
                cik="0000320193",
                filing_date=date(2023, 5, 4),
                accession_number="0000320193-23-000064",
                form="8-K",
                source_url=SUBMISSIONS_URL,
            ),
            EdgarFilingIndexEntry(
                cik="0000320193",
                filing_date=date(2022, 2, 1),
                accession_number="0000320193-22-000001",
                form="10-K",
                source_url=ARCHIVE_URL,
            ),
        )

    def test_rejects_reversed_window(self) -> None:
        with pytest.raises(ValueError, match="end"):
            _provider().filing_index(
                "0000320193",
                start=date(2024, 1, 2),
                end=date(2024, 1, 1),
            )


class TestFetch8KEvents:
    @respx.mock
    def test_extracts_registered_item_families_with_metadata(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        payload = {
            "filings": {
                "recent": {
                    "form": ["8-K", "8-K", "8-K", "8-K", "8-K", "10-Q"],
                    "filingDate": ["2023-08-03"] * 6,
                    "acceptanceDateTime": [
                        "2023-08-03T16:30:15.000Z",
                        "2023-08-03T17:00:00.000Z",
                        "2023-08-03T17:30:00.000Z",
                        "2023-08-03T18:00:00.000Z",
                        "2023-08-03T18:30:00.000Z",
                        "2023-08-03T19:00:00.000Z",
                    ],
                    "accessionNumber": ["a-1", "a-2", "a-3", "a-4", "a-5", "a-6"],
                    "items": [
                        "1.01,9.01",
                        "5.02,7.01",
                        "2.01",
                        "2.02,9.01",
                        "8.01,9.01",
                        "2.02",
                    ],
                    "primaryDocument": [
                        "a1.htm",
                        "a2.htm",
                        "a3.htm",
                        "a4.htm",
                        "a5.htm",
                        "q.htm",
                    ],
                    "primaryDocDescription": [
                        "Material agreement",
                        "Officer and Regulation FD disclosure",
                        "Acquisition",
                        "Results of operations",
                        "Other events",
                        "Quarterly report",
                    ],
                },
                "files": [],
            }
        }
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=payload))

        frame = _provider().fetch_8k_events(
            ["AAPL"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            item_types={"1.01", "2.02", "5.02", "7.01", "8.01"},
        )

        assert frame["accession_number"].to_list() == ["a-1", "a-2", "a-4", "a-5"]
        assert frame["matched_items"].to_list() == [
            ["1.01"],
            ["5.02", "7.01"],
            ["2.02"],
            ["8.01"],
        ]
        assert frame["primary_document"].to_list() == [
            "a1.htm",
            "a2.htm",
            "a4.htm",
            "a5.htm",
        ]
        assert frame["primary_document_description"].to_list() == [
            "Material agreement",
            "Officer and Regulation FD disclosure",
            "Results of operations",
            "Other events",
        ]

    @respx.mock
    def test_item_matching_uses_exact_tokens(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        payload = {
            "filings": {
                "recent": {
                    "form": ["8-K", "8-K"],
                    "filingDate": ["2023-08-03", "2023-08-03"],
                    "acceptanceDateTime": [
                        "2023-08-03T16:30:15.000Z",
                        "2023-08-03T17:00:00.000Z",
                    ],
                    "accessionNumber": ["a-1", "a-2"],
                    "items": ["12.02", "2.02"],
                },
                "files": [],
            }
        }
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=payload))

        frame = _provider().fetch_8k_events(
            ["AAPL"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            item_types={"2.02"},
        )

        assert frame["accession_number"].to_list() == ["a-2"]

    @respx.mock
    def test_empty_result_preserves_generic_schema(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        respx.get(SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=SUBMISSIONS_JSON))

        frame = _provider().fetch_8k_events(
            ["AAPL"],
            start=date(2010, 1, 1),
            end=date(2010, 12, 31),
        )

        assert frame.height == 0
        assert "matched_items" in frame.columns
        assert frame.schema["acceptance_datetime"] == pl.Datetime("us", "UTC")


class TestAccessionMetadata:
    def test_builds_archive_document_url_from_safe_basename(self) -> None:
        assert sec_archive_document_url(
            "0000002488",
            "0000002488-20-000006",
            "amdq4andfy2019earningsslides.pdf",
        ) == (
            "https://www.sec.gov/Archives/edgar/data/2488/"
            "000000248820000006/amdq4andfy2019earningsslides.pdf"
        )

        with pytest.raises(ValueError, match="SEC document filename must be a basename"):
            sec_archive_document_url(
                "0000002488",
                "0000002488-20-000006",
                "../action.pdf",
            )

    def test_complete_submission_url_is_bound_to_cik_and_accession(self) -> None:
        assert complete_submission_url(320193, "0000320193-23-000075") == COMPLETE_SUBMISSION_URL

    @pytest.mark.parametrize(
        ("cik", "accession"),
        [
            ("not-a-cik", "0000320193-23-000075"),
            ("0000320193", "../0000320193-23-000075"),
        ],
    )
    def test_complete_submission_url_rejects_invalid_identifiers(
        self,
        cik: str,
        accession: str,
    ) -> None:
        with pytest.raises(ValueError, match="CIK|accession"):
            complete_submission_url(cik, accession)

    def test_extracts_filing_symbol_and_exact_attachment_type(self) -> None:
        metadata = parse_edgar_accession_metadata(
            ACCESSION_SUBMISSION,
            cik="0000320193",
            accession_number="0000320193-23-000075",
        )

        assert metadata.cik == "0000320193"
        assert metadata.accession_number == "0000320193-23-000075"
        assert metadata.filing_symbols == ("AAPL",)
        assert metadata.has_ex_99_1 is True
        assert metadata.document_types == ("8-K", "EX-99.1")
        assert metadata.source_url == COMPLETE_SUBMISSION_URL

    def test_does_not_treat_nearby_exhibit_type_as_ex_99_1(self) -> None:
        metadata = parse_edgar_accession_metadata(
            ACCESSION_SUBMISSION.replace("<TYPE>EX-99.1\n", "<TYPE>EX-99.10\n"),
            cik="0000320193",
            accession_number="0000320193-23-000075",
        )

        assert metadata.has_ex_99_1 is False
        assert metadata.document_types == ("8-K", "EX-99.10")

    def test_missing_trading_symbol_is_explicit_not_backfilled(self) -> None:
        metadata = parse_edgar_accession_metadata(
            ACCESSION_SUBMISSION.replace(
                '<ix:nonNumeric name="dei:TradingSymbol" contextRef="c-1">AAPL</ix:nonNumeric>',
                "",
            ),
            cik="0000320193",
            accession_number="0000320193-23-000075",
        )

        assert metadata.filing_symbols == ()
        assert metadata.has_ex_99_1 is True

    @respx.mock
    def test_provider_fetches_one_accession_bound_source(self) -> None:
        route = respx.get(COMPLETE_SUBMISSION_URL).mock(
            return_value=httpx.Response(200, text=ACCESSION_SUBMISSION)
        )

        metadata = _provider().fetch_accession_metadata(
            "0000320193",
            "0000320193-23-000075",
        )

        assert metadata.filing_symbols == ("AAPL",)
        assert metadata.has_ex_99_1 is True
        assert route.call_count == 1
        assert route.calls.last.request.headers["User-Agent"] == "test test@example.com"


class TestErrors:
    @respx.mock
    def test_timeout_is_retried_then_succeeds(self) -> None:
        route = respx.get(TICKERS_URL).mock(
            side_effect=[
                httpx.ReadTimeout("slow SEC response"),
                httpx.Response(200, json=TICKERS_JSON),
            ]
        )

        assert _provider().resolve_cik("AAPL") == "0000320193"
        assert route.call_count == 2

    @respx.mock
    def test_server_error_is_retried_then_succeeds(self) -> None:
        route = respx.get(TICKERS_URL).mock(
            side_effect=[
                httpx.Response(503, json={}),
                httpx.Response(200, json=TICKERS_JSON),
            ]
        )

        assert _provider().resolve_cik("AAPL") == "0000320193"
        assert route.call_count == 2

    @respx.mock
    def test_client_error_is_not_retried(self) -> None:
        route = respx.get(TICKERS_URL).mock(return_value=httpx.Response(404, json={}))

        with pytest.raises(ProviderError, match="HTTP 404"):
            _provider().resolve_cik("AAPL")

        assert route.call_count == 1

    @respx.mock
    def test_429_raises_rate_limit_error(self) -> None:
        route = respx.get(TICKERS_URL).mock(return_value=httpx.Response(429, json={}))
        with pytest.raises(RateLimitError):
            _provider().resolve_cik("AAPL")
        assert route.call_count == 1

    @respx.mock
    def test_transport_error_wrapped_as_provider_error(self) -> None:
        route = respx.get(TICKERS_URL).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(ProviderError):
            _provider().resolve_cik("AAPL")
        assert route.call_count == 3

    @respx.mock
    def test_non_200_raises_provider_error(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(500, json={}))
        with pytest.raises(ProviderError):
            _provider().resolve_cik("AAPL")


class TestSettingsFactory:
    def test_factory_uses_settings_user_agent(self) -> None:
        settings = LiqDataSettings(sec_edgar_user_agent="ops ops@example.com")
        provider = create_sec_edgar_provider(settings)
        assert provider.name == "sec_edgar"

    def test_factory_raises_when_unconfigured(self) -> None:
        settings = LiqDataSettings(sec_edgar_user_agent=None)
        with pytest.raises(ValueError, match="SEC_EDGAR_USER_AGENT"):
            create_sec_edgar_provider(settings)


class TestEtiquette:
    def test_user_agent_header_is_sent(self) -> None:
        with respx.mock:
            route = respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
            _provider().resolve_cik("AAPL")
        assert route.calls.last.request.headers["User-Agent"] == "test test@example.com"

    def test_rate_limiter_configured_at_sec_cadence(self) -> None:
        assert _provider().rate_limiter.min_interval_seconds == pytest.approx(0.125)

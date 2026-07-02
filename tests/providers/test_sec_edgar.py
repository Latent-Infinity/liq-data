"""TDD pins for the SEC EDGAR filings provider.

Response payloads mirror the real EDGAR JSON contracts (company_tickers.json
and the submissions parallel-array shape) — filing metadata, not market data.
Transport is mocked with respx following the OANDA provider test pattern.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.providers.sec_edgar import SECEdgarProvider
from liq.data.settings import LiqDataSettings, create_sec_edgar_provider

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
ARCHIVE_URL = "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json"

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
    return SECEdgarProvider(user_agent="test test@example.com", **kwargs)


class TestConstruction:
    def test_requires_user_agent(self) -> None:
        with pytest.raises(ConfigurationError, match="user_agent"):
            SECEdgarProvider(user_agent="")

    def test_name(self) -> None:
        assert _provider().name == "sec_edgar"


class TestTickerToCik:
    @respx.mock
    def test_maps_ticker_to_padded_cik(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(200, json=TICKERS_JSON))
        provider = _provider()
        assert provider.resolve_cik("AAPL") == "0000320193"

    @respx.mock
    def test_hyphen_symbol_falls_back_to_dot_form(self) -> None:
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


class TestErrors:
    @respx.mock
    def test_429_raises_rate_limit_error(self) -> None:
        respx.get(TICKERS_URL).mock(return_value=httpx.Response(429, json={}))
        with pytest.raises(RateLimitError):
            _provider().resolve_cik("AAPL")

    @respx.mock
    def test_transport_error_wrapped_as_provider_error(self) -> None:
        respx.get(TICKERS_URL).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(ProviderError):
            _provider().resolve_cik("AAPL")

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

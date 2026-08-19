"""Tests for the FINRA short-interest provider (parse + point-in-time dissemination).

The committed sample is the header and first four records from FINRA's official
2024-12-31 consolidated file, downloaded from the provider URL on 2026-08-11.
"""

from __future__ import annotations

import hashlib
from datetime import date

import httpx
import polars as pl
import pytest

from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.providers.finra_short_interest import (
    SHORT_INTEREST_SCHEMA,
    FINRAShortInterestProvider,
    dissemination_date,
    parse_short_interest,
    settlement_dates,
    short_interest_url,
)

_SAMPLE = """accountingYearMonthNumber|symbolCode|issueName|issuerServicesGroupExchangeCode|marketClassCode|currentShortPositionQuantity|previousShortPositionQuantity|stockSplitFlag|averageDailyVolumeQuantity|daysToCoverQuantity|revisionFlag|changePercent|changePreviousNumber|settlementDate
20241231|A|Agilent Technologies Inc.|A|NYSE|2911097|3151214||1523270|1.91|Y|-7.62|-240117|2024-12-31
20241231|AA|Alcoa Corporation|A|NYSE|8240146|8328174||3445012|2.39||-1.06|-88028|2024-12-31
20241231|AAA|Alternative Access First Prior|E|ARCA|29509|19404||17436|1.69||52.08|10105|2024-12-31
20241231|AAAU|Goldman Sachs Physical Gold ET|H|BZX|643889|328481||1679326|1.00||96.02|315408|2024-12-31"""

_SETTLE = date(2024, 12, 31)
_DISSEM = date(2025, 1, 10)


def _parsed():
    return parse_short_interest(_SAMPLE, settlement_date=_SETTLE, dissemination_date=_DISSEM)


class TestParseShortInterest:
    def test_schema_and_row_count(self) -> None:
        df = _parsed()
        assert list(df.columns) == list(SHORT_INTEREST_SCHEMA)
        assert dict(df.schema) == dict(SHORT_INTEREST_SCHEMA)
        assert df.height == 4

    def test_values_and_numeric_casts(self) -> None:
        df = _parsed().filter(pl.col("symbol") == "A")
        row = df.row(0, named=True)
        assert row["short_interest"] == 2911097
        assert row["prev_short_interest"] == 3151214
        assert row["avg_daily_volume"] == 1523270
        assert row["days_to_cover"] == pytest.approx(1.91)
        assert row["change_pct"] == pytest.approx(-7.62)
        assert row["change_shares"] == -240117
        assert row["market"] == "NYSE"
        assert row["revision_flag"] == "Y"

    def test_keyed_by_supplied_dates_and_sorted(self) -> None:
        df = _parsed()
        assert df["settlement_date"].unique().to_list() == [_SETTLE]
        assert df["dissemination_date"].unique().to_list() == [_DISSEM]
        assert df["symbol"].to_list() == sorted(df["symbol"].to_list())

    def test_missing_required_columns_are_rejected(self) -> None:
        with pytest.raises(ProviderError, match="missing required columns"):
            parse_short_interest(
                "symbolCode|issueName\nA|Agilent Technologies Inc.\n",
                settlement_date=_SETTLE,
                dissemination_date=_DISSEM,
            )


class TestDisseminationDate:
    @pytest.mark.parametrize(
        ("settle", "expected"),
        [
            (date(2026, 1, 15), date(2026, 1, 27)),  # FINRA worked example (skips MLK)
            (date(2026, 1, 30), date(2026, 2, 10)),  # FINRA worked example
        ],
    )
    def test_matches_finra_published_schedule(self, settle: date, expected: date) -> None:
        assert dissemination_date(settle) == expected

    def test_is_seven_sessions_after_settlement(self) -> None:
        assert dissemination_date(_SETTLE) > _SETTLE


class TestSettlementDates:
    def test_two_per_month_mid_and_month_end(self) -> None:
        dates = settlement_dates(date(2026, 1, 1), date(2026, 3, 31))
        assert dates == [
            date(2026, 1, 15),
            date(2026, 1, 30),
            date(2026, 2, 13),  # Feb 15 is a Sunday -> prior session Fri Feb 13
            date(2026, 2, 27),  # Feb 28 is a Saturday -> Fri Feb 27
            date(2026, 3, 13),  # Mar 15 is a Sunday -> Fri Mar 13
            date(2026, 3, 31),
        ]

    def test_respects_range_bounds(self) -> None:
        dates = settlement_dates(date(2026, 1, 20), date(2026, 2, 20))
        assert min(dates) >= date(2026, 1, 20) and max(dates) <= date(2026, 2, 20)


class TestProviderConstruction:
    def test_official_archive_url_is_date_keyed(self) -> None:
        assert short_interest_url(_SETTLE) == (
            "https://cdn.finra.org/equity/otcmarket/biweekly/shrt20241231.csv"
        )

    def test_fetch_preserves_source_file_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = FINRAShortInterestProvider(
            "Latent Infinity research@example.com", min_interval_seconds=0
        )
        monkeypatch.setattr(provider, "_get_text", lambda _url: _SAMPLE)

        frame = provider.fetch_short_interest(_SETTLE)

        assert frame["source_file_url"].unique().to_list() == [short_interest_url(_SETTLE)]
        assert frame["source_file_sha256"].unique().to_list() == [
            hashlib.sha256(_SAMPLE.encode()).hexdigest()
        ]
        assert frame["revision_flag"].to_list() == ["Y", None, None, None]

    def test_requires_contact_user_agent(self) -> None:
        with pytest.raises(ConfigurationError):
            FINRAShortInterestProvider("")

    def test_accepts_user_agent(self) -> None:
        p = FINRAShortInterestProvider("Latent Infinity research@example.com")
        assert p.name == "finra_short_interest"

    @pytest.mark.parametrize(
        ("status_code", "error_type"),
        [(429, RateLimitError), (503, ProviderError)],
    )
    def test_http_status_errors_are_mapped(
        self, status_code: int, error_type: type[Exception]
    ) -> None:
        provider = FINRAShortInterestProvider(
            "Latent Infinity research@example.com", min_interval_seconds=0
        )
        provider._client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status_code, request=request)
            )
        )
        with pytest.raises(error_type):
            provider._get_text("https://example.invalid/short-interest.csv")

    def test_transport_errors_are_mapped(self) -> None:
        def fail(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        provider = FINRAShortInterestProvider(
            "Latent Infinity research@example.com", min_interval_seconds=0
        )
        provider._client = httpx.Client(transport=httpx.MockTransport(fail))
        with pytest.raises(ProviderError, match="FINRA request failed"):
            provider._get_text("https://example.invalid/short-interest.csv")

"""S&P 500 membership reference adapter.

Parser mechanics run on small constructed tables; the Wikipedia parser is
additionally pinned on a verbatim excerpt of the live changes-table wikitext
(retrieved 2026-07-02) so real formatting variants — single-line and
multi-line rows, wiki-links with display pipes, refs — stay covered.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from liq.data.exceptions import ConfigurationError
from liq.data.providers.sp500_membership import (
    SNAPSHOT_CSV_URL,
    WIKIPEDIA_API_URL,
    CrossCheckReport,
    SP500MembershipProvider,
    annotate_deltas_with_wikipedia_cross_check,
    build_membership_deltas,
    cross_check_deltas,
    parse_wikipedia_changes,
    parse_wikipedia_sectors,
)

_UA = "liq-research/0.1 (test@example.com)"

_SNAPSHOT_CSV = (
    'date,tickers\n2005-01-03,"AAA,BBB,CCC"\n2005-02-01,"AAA,BBB,DDD"\n2005-03-01,"AAA,BBB,DDD"\n'
)

# Verbatim excerpt of the live "Selected changes" wikitext, 2026-07-02.
_WIKI_EXCERPT = """{|  class="wikitable sortable" id="changes"
|-
! data-sort-type="date" rowspan="2" | Effective Date
! colspan="2" | Added
! colspan="2" | Removed
! rowspan="2" | Reason
|-
! Ticker || Security || Ticker || Security
|-
|June 30, 2026 ||  ||  || CAG || [[Conagra Brands]] || Market capitalization change.<ref name="sp20260623">{{cite web |url=https://example.test |title=X |date=June 23, 2026}}</ref>
|-
|June 29, 2026 || HONA || [[Honeywell Aerospace]] || || || S&P 500 constituent [[Honeywell]] spun off Honeywell Aerospace.<ref name="sp20260623" />
|-
|June 22, 2026
|MRVL
|[[Marvell Technology]]
|POOL
|[[Pool Corporation]]
|Market capitalization change.<ref name=":5">{{Cite web |date=2026-06-05 |title=Y |url=https://example.test}}</ref>
|-
|May 7, 2026
|VEEV
|[[Veeva Systems]]
|CTRA
|[[Coterra|Coterra Energy]]
|S&P 500 constituent [[Devon Energy|Devon Energy Corp.]] acquired Coterra Energy.<ref>{{Cite web |title=Z |url=https://example.test}}</ref>
|}"""


class TestProviderConstruction:
    def test_requires_user_agent(self) -> None:
        with pytest.raises(ConfigurationError):
            SP500MembershipProvider(user_agent="")


class TestSnapshotFetch:
    @respx.mock
    def test_fetch_membership_snapshots(self) -> None:
        respx.get(SNAPSHOT_CSV_URL).mock(return_value=httpx.Response(200, text=_SNAPSHOT_CSV))
        provider = SP500MembershipProvider(user_agent=_UA)
        df = provider.fetch_membership_snapshots()
        assert df.height == 3
        assert df["date"].to_list() == [date(2005, 1, 3), date(2005, 2, 1), date(2005, 3, 1)]
        assert df["tickers"].to_list()[0] == ["AAA", "BBB", "CCC"]

    @respx.mock
    def test_fetch_error_raises(self) -> None:
        respx.get(SNAPSHOT_CSV_URL).mock(return_value=httpx.Response(500))
        provider = SP500MembershipProvider(user_agent=_UA)
        with pytest.raises(httpx.HTTPStatusError):
            provider.fetch_membership_snapshots()


class TestDeltaBuilder:
    def test_deltas_from_consecutive_snapshots(self) -> None:
        import polars as pl

        snapshots = pl.DataFrame(
            {
                "date": [date(2005, 1, 3), date(2005, 2, 1), date(2005, 3, 1)],
                "tickers": [["AAA", "BBB", "CCC"], ["AAA", "BBB", "DDD"], ["AAA", "BBB", "DDD"]],
            }
        )
        deltas = build_membership_deltas(snapshots)
        rows = sorted(deltas.rows(named=True), key=lambda r: (str(r["date"]), r["symbol"]))
        assert [(r["date"], r["symbol"], r["action"]) for r in rows] == [
            (date(2005, 2, 1), "CCC", "removed"),
            (date(2005, 2, 1), "DDD", "added"),
        ]
        # The unchanged 2005-03-01 snapshot must contribute no deltas.
        assert deltas.height == 2

    def test_prev_date_recorded_for_provenance(self) -> None:
        import polars as pl

        snapshots = pl.DataFrame(
            {
                "date": [date(2005, 1, 3), date(2005, 2, 1)],
                "tickers": [["AAA"], ["BBB"]],
            }
        )
        deltas = build_membership_deltas(snapshots)
        assert set(deltas["prev_snapshot_date"].to_list()) == {date(2005, 1, 3)}


class TestWikipediaParser:
    def test_parses_all_row_styles(self) -> None:
        changes = parse_wikipedia_changes(_WIKI_EXCERPT)
        rows = {(r["date"], r["symbol"], r["action"]) for r in changes.rows(named=True)}
        assert (date(2026, 6, 30), "CAG", "removed") in rows
        assert (date(2026, 6, 29), "HONA", "added") in rows
        assert (date(2026, 6, 22), "MRVL", "added") in rows
        assert (date(2026, 6, 22), "POOL", "removed") in rows
        assert (date(2026, 5, 7), "VEEV", "added") in rows
        assert (date(2026, 5, 7), "CTRA", "removed") in rows
        # Header rows and empty cells never produce entries.
        assert all(r["symbol"].isupper() for r in changes.rows(named=True))

    @respx.mock
    def test_fetch_wikipedia_changes(self) -> None:
        payload = {"parse": {"wikitext": _WIKI_EXCERPT}}
        respx.get(WIKIPEDIA_API_URL).mock(return_value=httpx.Response(200, json=payload))
        provider = SP500MembershipProvider(user_agent=_UA)
        df = provider.fetch_wikipedia_changes()
        assert df.height == 6


class TestCrossCheck:
    def test_all_matched_within_window(self) -> None:
        import polars as pl

        deltas = pl.DataFrame(
            {
                "date": [date(2026, 6, 25)],
                "symbol": ["MRVL"],
                "action": ["added"],
                "prev_snapshot_date": [date(2026, 6, 18)],
            }
        )
        wiki = pl.DataFrame({"date": [date(2026, 6, 22)], "symbol": ["MRVL"], "action": ["added"]})
        report = cross_check_deltas(deltas, wiki, window_days=7)
        assert isinstance(report, CrossCheckReport)
        assert report.n_wiki_in_window == 1
        assert report.n_unmatched == 0
        assert report.mismatch_rate == 0.0

    def test_unmatched_wiki_change_counts_as_mismatch(self) -> None:
        import polars as pl

        deltas = pl.DataFrame(
            {
                "date": [date(2026, 6, 25)],
                "symbol": ["MRVL"],
                "action": ["added"],
                "prev_snapshot_date": [date(2026, 6, 18)],
            }
        )
        wiki = pl.DataFrame(
            {
                "date": [date(2026, 6, 22), date(2026, 6, 22)],
                "symbol": ["MRVL", "POOL"],
                "action": ["added", "removed"],
            }
        )
        report = cross_check_deltas(deltas, wiki, window_days=7)
        assert report.n_wiki_in_window == 2
        assert report.n_unmatched == 1
        assert report.mismatch_rate == pytest.approx(0.5)
        assert report.unmatched[0]["symbol"] == "POOL"

    def test_wiki_rows_outside_delta_window_excluded(self) -> None:
        import polars as pl

        deltas = pl.DataFrame(
            {
                "date": [date(2026, 6, 25)],
                "symbol": ["MRVL"],
                "action": ["added"],
                "prev_snapshot_date": [date(2026, 6, 18)],
            }
        )
        wiki = pl.DataFrame({"date": [date(1999, 1, 4)], "symbol": ["OLD"], "action": ["removed"]})
        report = cross_check_deltas(deltas, wiki, window_days=7)
        assert report.n_wiki_in_window == 0
        assert report.mismatch_rate == 0.0

    def test_delta_annotation_requires_date_window_match(self) -> None:
        import polars as pl

        deltas = pl.DataFrame(
            {
                "date": [date(2026, 6, 25), date(2026, 8, 1)],
                "symbol": ["MRVL", "MRVL"],
                "action": ["added", "added"],
                "prev_snapshot_date": [date(2026, 6, 18), date(2026, 7, 25)],
            }
        )
        wiki = pl.DataFrame({"date": [date(2026, 6, 22)], "symbol": ["MRVL"], "action": ["added"]})

        annotated = annotate_deltas_with_wikipedia_cross_check(deltas, wiki, window_days=7)

        assert annotated["wikipedia_cross_check"].to_list() == [
            "confirmed",
            "not-in-wikipedia-window",
        ]


_CONSTITUENTS_EXCERPT = """{| class="wikitable sortable sticky-header" id="constituents"
|-
![[Ticker symbol|Symbol]]
! Security !! [[Global Industry Classification Standard|GICS]] Sector !! GICS Sub-Industry !! Headquarters Location !! Date added !! [[Central Index Key|CIK]] !! Founded
|-
|{{NyseSymbol|MMM}}
|[[3M]]|| Industrials || Industrial Conglomerates || [[Saint Paul, Minnesota]] || 1957-03-04 || 0000066740 || 1902
|-
|{{NyseSymbol|ABT}}
|[[Abbott Laboratories]]|| Health Care || Health Care Equipment || [[North Chicago, Illinois]] || 1957-03-04 || 0000001800 || 1888
|-
|{{NasdaqSymbol|ACN}}
|[[Accenture]]|| Information Technology || IT Consulting & Other Services || [[Dublin]], Ireland || 2011-07-06 || 0001467373 || 1989
|}"""


class TestSectorParser:
    def test_parses_symbol_to_gics_sector(self) -> None:
        sectors = parse_wikipedia_sectors(_CONSTITUENTS_EXCERPT)
        assert sectors["MMM"] == "Industrials"
        assert sectors["ABT"] == "Health Care"
        assert sectors["ACN"] == "Information Technology"
        assert len(sectors) == 3

    def test_parses_single_line_constituent_row(self) -> None:
        excerpt = """{| class="wikitable sortable sticky-header" id="constituents"
|-
! Symbol !! Security !! GICS Sector
|-
|{{NyseSymbol|MMM}} || [[3M]] || Industrials
|}"""
        assert parse_wikipedia_sectors(excerpt)["MMM"] == "Industrials"

    @respx.mock
    def test_fetch_wikipedia_sectors(self) -> None:
        payload = {"parse": {"wikitext": _CONSTITUENTS_EXCERPT}}
        respx.get(WIKIPEDIA_API_URL).mock(return_value=httpx.Response(200, json=payload))
        provider = SP500MembershipProvider(user_agent=_UA)
        assert provider.fetch_wikipedia_sectors()["MMM"] == "Industrials"

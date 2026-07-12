"""S&P 500 historical membership reference adapter.

Two independent public sources back point-in-time membership work:

* **Snapshots** — the community-maintained ``fja05680/sp500`` dataset
  (``date,tickers`` CSV of full index composition since 1996), fetched from
  GitHub raw. Consecutive snapshots diff into membership deltas.
* **Cross-check** — Wikipedia's "Selected changes" table on the
  *List of S&P 500 companies* page, fetched as wikitext through the MediaWiki
  API and parsed without HTML dependencies.

This is a reference/event adapter in the same standalone category as
``SECEdgarProvider`` — it does not implement the OHLCV ``BaseProvider``
contract. Reconstruction quality is judged by ``cross_check_deltas``: the
fraction of Wikipedia-listed changes (inside the reconstruction window) that
have no matching delta is the mismatch rate the caller gates on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx
import polars as pl

from liq.data.exceptions import ConfigurationError

SNAPSHOT_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
# NOTE: the repo's non-"(Updated)" CSV is stale (ends 2019-01). The dataset is
# entity-PIT but NOT ticker-PIT: later tickers are back-applied to history
# (e.g. Lehman 2008 appears as LEHMQ, Priceline 2009 as BKNG), so
# contemporaneous-ticker sources will disagree on renames/class changes.
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
_WIKIPEDIA_PAGE = "List of S&P 500 companies"

_TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")
# [[target|display]] → display; [[target]] → target
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]")
_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL)
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)


class SP500MembershipProvider:
    """Fetches the two public S&P 500 membership sources."""

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not user_agent:
            raise ConfigurationError(
                "S&P 500 membership sources require a contact user_agent (fair-use policy)"
            )
        self._user_agent = user_agent
        self._timeout = timeout_seconds
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def fetch_membership_snapshots(self) -> pl.DataFrame:
        """Full-composition snapshots: ``date`` (pl.Date), ``tickers`` (list[str])."""
        response = self._get_client().get(SNAPSHOT_CSV_URL)
        response.raise_for_status()
        raw = pl.read_csv(response.content)
        return raw.with_columns(
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("tickers").str.split(",").list.eval(pl.element().str.strip_chars()),
        ).sort("date")

    def fetch_wikipedia_changes(self) -> pl.DataFrame:
        """Long-form change entries parsed from the live changes table."""
        response = self._get_client().get(
            WIKIPEDIA_API_URL,
            params={
                "action": "parse",
                "page": _WIKIPEDIA_PAGE,
                "prop": "wikitext",
                "format": "json",
                "formatversion": "2",
            },
        )
        response.raise_for_status()
        wikitext = response.json()["parse"]["wikitext"]
        return parse_wikipedia_changes(wikitext)

    def fetch_wikipedia_sectors(self) -> dict[str, str]:
        """Current ``symbol -> GICS sector`` map from the constituents table."""
        response = self._get_client().get(
            WIKIPEDIA_API_URL,
            params={
                "action": "parse",
                "page": _WIKIPEDIA_PAGE,
                "prop": "wikitext",
                "format": "json",
                "formatversion": "2",
            },
        )
        response.raise_for_status()
        return parse_wikipedia_sectors(response.json()["parse"]["wikitext"])


_SYMBOL_TEMPLATE = re.compile(r"\{\{(?:Nyse|Nasdaq|Cboe)[Ss]ymbol\|([A-Z][A-Z0-9.\-]{0,6})\}\}")


def parse_wikipedia_sectors(wikitext: str) -> dict[str, str]:
    """``symbol -> GICS sector`` from the current-constituents table.

    Current-membership sectors only; historical sector reassignments are not
    reconstructed (an accepted screen-grade caveat alongside entity-PIT).
    """
    _, found, tail = wikitext.partition('id="constituents"')
    if not found:
        raise ValueError('wikitext has no constituents table (id="constituents")')
    table = tail.partition("|}")[0]
    sectors: dict[str, str] = {}
    for block in table.split("\n|-"):
        symbol_match = _SYMBOL_TEMPLATE.search(block)
        if symbol_match is None:
            continue
        cells: list[str] = []
        symbol_cell_index: int | None = None
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("|") or line.startswith("|}"):
                continue
            for cell in line.lstrip("|").split("||"):
                if _SYMBOL_TEMPLATE.search(cell) is not None:
                    symbol_cell_index = len(cells)
                cells.append(_WIKI_LINK.sub(r"\1", cell).strip())
        if symbol_cell_index is None:
            continue
        sector_index = symbol_cell_index + 2  # symbol, security, GICS sector
        if sector_index < len(cells) and cells[sector_index]:
            sectors[symbol_match.group(1)] = cells[sector_index]
    return sectors


def parse_wikipedia_changes(wikitext: str) -> pl.DataFrame:
    """Parse the ``id="changes"`` table into ``date / symbol / action`` rows.

    Handles both single-line (``|date || T || [[name]] || ...``) and
    multi-line (one leading-pipe cell per line) row styles; refs, templates,
    and wiki-link display pipes are stripped before cell splitting.
    """
    _, found, tail = wikitext.partition('id="changes"')
    if not found:
        raise ValueError('wikitext has no changes table (id="changes")')
    table = tail.partition("|}")[0]
    table = _REF.sub("", table)
    table = _TEMPLATE.sub("", table)
    table = _WIKI_LINK.sub(r"\1", table)

    entries: list[dict[str, object]] = []
    for block in table.split("\n|-"):
        cells: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("|") or line.startswith("|}"):
                continue
            cells.extend(cell.strip() for cell in line.lstrip("|").split("||"))
        if not cells:
            continue
        try:
            effective = datetime.strptime(cells[0], "%B %d, %Y").date()
        except ValueError:
            continue  # header or malformed row
        for index, action in ((1, "added"), (3, "removed")):
            if index < len(cells) and _TICKER.match(cells[index]):
                entries.append({"date": effective, "symbol": cells[index], "action": action})

    return pl.DataFrame(
        entries,
        schema={"date": pl.Date, "symbol": pl.String, "action": pl.String},
    )


def build_membership_deltas(snapshots: pl.DataFrame) -> pl.DataFrame:
    """Diff consecutive snapshots into add/remove deltas with provenance."""
    entries: list[dict[str, object]] = []
    rows = snapshots.sort("date").rows(named=True)
    for previous, current in zip(rows, rows[1:], strict=False):
        before, after = set(previous["tickers"]), set(current["tickers"])
        for symbol in sorted(after - before):
            entries.append(
                {
                    "date": current["date"],
                    "symbol": symbol,
                    "action": "added",
                    "prev_snapshot_date": previous["date"],
                }
            )
        for symbol in sorted(before - after):
            entries.append(
                {
                    "date": current["date"],
                    "symbol": symbol,
                    "action": "removed",
                    "prev_snapshot_date": previous["date"],
                }
            )
    return pl.DataFrame(
        entries,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "action": pl.String,
            "prev_snapshot_date": pl.Date,
        },
    )


@dataclass(frozen=True)
class CrossCheckReport:
    """Cross-source agreement between deltas and Wikipedia change entries."""

    n_wiki_in_window: int
    n_matched: int
    n_unmatched: int
    mismatch_rate: float
    unmatched: list[dict[str, object]]


def cross_check_deltas(
    deltas: pl.DataFrame,
    wiki_changes: pl.DataFrame,
    *,
    window_days: int = 7,
) -> CrossCheckReport:
    """Match Wikipedia change entries against reconstructed deltas.

    A Wikipedia entry matches when a delta with the same symbol and action
    lies within ``window_days`` of its effective date (snapshots trail
    effective dates by up to one snapshot interval). Only Wikipedia entries
    inside the reconstruction's date span are judged.
    """
    if deltas.is_empty():
        return CrossCheckReport(0, 0, 0, 0.0, [])
    window = timedelta(days=window_days)
    span_start = deltas["date"].min()
    span_end = deltas["date"].max()
    assert isinstance(span_start, date) and isinstance(span_end, date)

    delta_index: dict[tuple[str, str], list[date]] = {}
    for row in deltas.rows(named=True):
        delta_index.setdefault((row["symbol"], row["action"]), []).append(row["date"])

    in_window = 0
    unmatched: list[dict[str, object]] = []
    for row in wiki_changes.rows(named=True):
        effective = row["date"]
        if effective < span_start - window or effective > span_end + window:
            continue
        in_window += 1
        candidates = delta_index.get((row["symbol"], row["action"]), [])
        if not any(abs(d - effective) <= window for d in candidates):
            unmatched.append(dict(row))

    matched = in_window - len(unmatched)
    rate = (len(unmatched) / in_window) if in_window else 0.0
    return CrossCheckReport(
        n_wiki_in_window=in_window,
        n_matched=matched,
        n_unmatched=len(unmatched),
        mismatch_rate=rate,
        unmatched=unmatched,
    )


def annotate_deltas_with_wikipedia_cross_check(
    deltas: pl.DataFrame,
    wiki_changes: pl.DataFrame,
    *,
    window_days: int = 7,
) -> pl.DataFrame:
    """Attach date-window-aware Wikipedia confirmation to each delta row.

    ``confirmed`` means a Wikipedia selected-change entry with the same symbol
    and action lies within ``window_days`` of the snapshot-derived delta date.
    This is intentionally stricter than a symbol/action-ever-seen check: a
    ticker can enter, leave, and re-enter the index over time.
    """
    window = timedelta(days=window_days)
    wiki_index: dict[tuple[str, str], list[date]] = {}
    for row in wiki_changes.rows(named=True):
        wiki_index.setdefault((row["symbol"], row["action"]), []).append(row["date"])

    return deltas.with_columns(
        pl.struct(["date", "symbol", "action"])
        .map_elements(
            lambda row: (
                "confirmed"
                if any(
                    abs(candidate - row["date"]) <= window
                    for candidate in wiki_index.get((row["symbol"], row["action"]), [])
                )
                else "not-in-wikipedia-window"
            ),
            return_dtype=pl.String,
        )
        .alias("wikipedia_cross_check")
    )

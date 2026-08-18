"""FINRA consolidated equity short-interest provider (free, reference data — not OHLCV).

Fetches FINRA's semi-monthly consolidated short-interest file from the anonymous CDN
(``https://cdn.finra.org/equity/otcmarket/biweekly/shrt{YYYYMMDD}.csv`` — pipe-delimited, whole
universe, no API key). Point-in-time correct: each record carries a ``dissemination_date`` (the 7th
XNYS session after settlement — when the aggregate became public), distinct from ``settlement_date``.

Deliberately NOT a :class:`~liq.data.providers.base.BaseProvider` subclass: that ABC's contract is
OHLCV bars, which a settlement-date cross-section cannot honestly implement (interface segregation) —
same rationale as ``sec_edgar.py``/``sp500_membership.py``. FINRA fair-use: a descriptive contact
``User-Agent``; one bulk file per settlement date (never a per-symbol loop).

Coverage note: files are consolidated (exchange-listed + OTC) only from **June 2021**; earlier files
are OTC-only. The current cadence is semi-monthly (mid-month + month-end); the
settlement-to-dissemination logic is date-driven rather than tied to a file count.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import httpx
import polars as pl

from liq.data.calendar import _calendar
from liq.data.exceptions import ConfigurationError, ProviderError, RateLimitError
from liq.data.rate_limiter import RateLimiter

_CDN_URL = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{yyyymmdd}.csv"
_DISSEMINATION_SESSIONS = 7  # FINRA publishes on the 7th business day after settlement

#: Normalized output schema (one row per symbol × settlement date).
SHORT_INTEREST_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "settlement_date": pl.Date,
    "dissemination_date": pl.Date,
    "symbol": pl.String,
    "issue_name": pl.String,
    "market": pl.String,
    "short_interest": pl.Int64,
    "prev_short_interest": pl.Int64,
    "avg_daily_volume": pl.Int64,
    "days_to_cover": pl.Float64,
    "revision_flag": pl.String,
    "change_pct": pl.Float64,
    "change_shares": pl.Int64,
}

# Raw FINRA column -> normalized name (the current consolidated 14-column layout).
_COLUMN_MAP = {
    "symbolCode": "symbol",
    "issueName": "issue_name",
    "marketClassCode": "market",
    "currentShortPositionQuantity": "short_interest",
    "previousShortPositionQuantity": "prev_short_interest",
    "averageDailyVolumeQuantity": "avg_daily_volume",
    "daysToCoverQuantity": "days_to_cover",
    "revisionFlag": "revision_flag",
    "changePercent": "change_pct",
    "changePreviousNumber": "change_shares",
}
_INT_COLS = ("short_interest", "prev_short_interest", "avg_daily_volume", "change_shares")


def dissemination_date(settlement_date: date) -> date:
    """The 7th XNYS session after ``settlement_date`` — when FINRA disseminates the aggregate."""
    cal = _calendar()
    anchor = cal.date_to_session(settlement_date.isoformat(), direction="previous")
    # sessions_window includes the anchor, so +1 elements gives the Nth session AFTER it.
    window = cal.sessions_window(anchor, _DISSEMINATION_SESSIONS + 1)
    return window[-1].date()


def settlement_dates(start: date, end: date) -> list[date]:
    """FINRA semi-monthly settlement dates in ``[start, end]``: the 15th (or the prior XNYS
    session) and each month's last XNYS session. Sorted, deduped — these key the CDN filenames."""
    cal = _calendar()
    out: list[date] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        mid = cal.date_to_session(date(year, month, 15).isoformat(), direction="previous").date()
        last_cal = (
            date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
        )
        eom = cal.date_to_session(last_cal.isoformat(), direction="previous").date()
        for candidate in (mid, eom):
            if start <= candidate <= end and candidate not in out:
                out.append(candidate)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return sorted(out)


def parse_short_interest(
    text: str, *, settlement_date: date, dissemination_date: date
) -> pl.DataFrame:
    """Parse one pipe-delimited FINRA short-interest file into the normalized schema.

    Pure and deterministic. ``days_to_cover`` of ``N/A`` (ADV = 0) becomes null; all other numerics
    are cast (a bad numeric ⇒ null, never imputed). Rows are keyed by the supplied dates, not the
    file's redundant date columns.
    """
    raw = pl.read_csv(io.StringIO(text), separator="|", infer_schema_length=0)
    missing = sorted(set(_COLUMN_MAP).difference(raw.columns))
    if missing:
        raise ProviderError(f"FINRA file missing required columns: {', '.join(missing)}")
    frame = raw.select(list(_COLUMN_MAP)).rename(_COLUMN_MAP)
    frame = frame.with_columns(
        pl.col("days_to_cover").replace("N/A", None).cast(pl.Float64, strict=False),
        pl.col("change_pct").cast(pl.Float64, strict=False),
        *[pl.col(c).cast(pl.Int64, strict=False) for c in _INT_COLS if c in frame.columns],
    ).with_columns(
        settlement_date=pl.lit(settlement_date, dtype=pl.Date),
        dissemination_date=pl.lit(dissemination_date, dtype=pl.Date),
    )
    return frame.select(list(SHORT_INTEREST_SCHEMA)).sort("symbol")


class FINRAShortInterestProvider:
    """Consolidated equity short-interest over the free FINRA CDN files."""

    name = "finra_short_interest"

    def __init__(
        self,
        user_agent: str,
        *,
        timeout: float = 30.0,
        min_interval_seconds: float = 0.5,
    ) -> None:
        if not user_agent:
            raise ConfigurationError(
                "FINRA short interest requires a contact user_agent (fair-use etiquette)"
            )
        self._user_agent = user_agent
        self._timeout = timeout
        self.rate_limiter = RateLimiter(min_interval_seconds=min_interval_seconds)
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:  # pragma: no cover - live transport
        if self._client is None:
            self._client = httpx.Client(
                headers={"User-Agent": self._user_agent, "Accept-Encoding": "gzip,deflate"},
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    def _get_text(self, url: str) -> str:  # pragma: no cover - live transport
        self.rate_limiter.acquire()
        try:
            response = self._get_client().get(url)
        except httpx.RequestError as exc:
            raise ProviderError(f"FINRA request failed: {exc}") from exc
        if response.status_code == 429:
            raise RateLimitError("FINRA rate limit exceeded")
        if response.status_code != 200:
            raise ProviderError(f"FINRA returned HTTP {response.status_code} for {url}")
        return response.text

    def fetch_short_interest(
        self, settlement_date: date
    ) -> pl.DataFrame:  # pragma: no cover - live
        """Fetch + normalize the consolidated short-interest file for one settlement date."""
        url = _CDN_URL.format(yyyymmdd=settlement_date.strftime("%Y%m%d"))
        text = self._get_text(url)
        return parse_short_interest(
            text,
            settlement_date=settlement_date,
            dissemination_date=dissemination_date(settlement_date),
        )

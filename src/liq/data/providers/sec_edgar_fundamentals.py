"""SEC EDGAR XBRL fundamentals provider (financial statements, not OHLCV).

Fetches the free ``companyfacts`` XBRL endpoint and normalizes it into a
point-in-time :class:`FundamentalsSnapshot` via
:mod:`liq.data.fundamentals.parser`. CIK resolution reuses the tested
:class:`SECEdgarProvider` ticker map; only the companyfacts transport is
added here (a small, self-contained client) so this provider keeps its own
narrow identity without altering the event provider.

SEC fair-use etiquette: a real contact ``User-Agent`` is required and
requests are throttled via :class:`RateLimiter`.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from liq.data.exceptions import (
    ConfigurationError,
    ProviderError,
    ProviderNoDataError,
    RateLimitError,
)
from liq.data.fundamentals.models import FundamentalsSnapshot
from liq.data.fundamentals.parser import build_snapshot
from liq.data.providers.sec_edgar import SECEdgarProvider
from liq.data.rate_limiter import RateLimiter

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


class SECEdgarFundamentalsProvider:
    """PIT fundamentals over the free SEC EDGAR ``companyfacts`` endpoint."""

    name = "sec_edgar_fundamentals"

    def __init__(
        self,
        user_agent: str,
        *,
        timeout: float = 30.0,
        min_interval_seconds: float = 0.125,
        cik_resolver: SECEdgarProvider | None = None,
    ) -> None:
        if not user_agent:
            raise ConfigurationError("SEC EDGAR requires a contact user_agent (fair-use policy)")
        self._user_agent = user_agent
        self._timeout = timeout
        self.rate_limiter = RateLimiter(min_interval_seconds=min_interval_seconds)
        self._client: httpx.Client | None = None
        self._cik_resolver = cik_resolver or SECEdgarProvider(
            user_agent, timeout=timeout, min_interval_seconds=min_interval_seconds
        )

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
        if response.status_code == 404:
            raise ProviderNoDataError(f"SEC EDGAR returned HTTP 404 for {url}")
        if response.status_code != 200:
            raise ProviderError(f"SEC EDGAR returned HTTP {response.status_code} for {url}")
        return response.json()

    # -- public ------------------------------------------------------------
    def resolve_cik(self, symbol: str) -> str | None:
        """Zero-padded CIK for a ticker (delegates to the event provider)."""
        return self._cik_resolver.resolve_cik(symbol)

    def fetch_companyfacts(self, cik10: str) -> dict[str, Any]:
        """Raw companyfacts JSON for a zero-padded CIK."""
        return self._get_json(COMPANYFACTS_URL.format(cik10=cik10))

    def fetch_fundamentals(self, symbol: str, *, as_of: date) -> FundamentalsSnapshot | None:
        """PIT fundamentals for ``symbol`` as of ``as_of``.

        Returns ``None`` when the ticker has no known CIK (caller records an
        exclusion). Only filings dated on or before ``as_of`` are used.
        """
        cik10 = self.resolve_cik(symbol)
        if cik10 is None:
            return None
        companyfacts = self.fetch_companyfacts(cik10)
        return build_snapshot(symbol, cik10, companyfacts, as_of)


__all__ = ["COMPANYFACTS_URL", "SECEdgarFundamentalsProvider"]

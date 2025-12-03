"""Provider policy definitions for rate limits, delays, and corp-action stance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPolicy:
    """Static provider policy metadata."""

    requests_per_minute: int | None = None
    burst: int | None = None
    delayed_feed: bool = False
    corporate_actions: str | None = None  # e.g., "adjusted", "unadjusted", "provider-handled"


# Baseline policy map; extend as needed
POLICIES: dict[str, ProviderPolicy] = {
    "alpaca": ProviderPolicy(requests_per_minute=200, burst=200, delayed_feed=True, corporate_actions="adjusted"),
    "interactive_brokers": ProviderPolicy(requests_per_minute=6, burst=6, delayed_feed=False, corporate_actions="provider-handled"),
    "oanda": ProviderPolicy(requests_per_minute=None, burst=None, delayed_feed=False, corporate_actions="provider-handled"),
    "coinbase": ProviderPolicy(requests_per_minute=None, burst=None, delayed_feed=False, corporate_actions="provider-handled"),
    "binance": ProviderPolicy(requests_per_minute=None, burst=None, delayed_feed=False, corporate_actions="provider-handled"),
    "polygon": ProviderPolicy(requests_per_minute=None, burst=None, delayed_feed=False, corporate_actions="adjusted"),
    "tradestation": ProviderPolicy(requests_per_minute=120, burst=120, delayed_feed=False, corporate_actions="provider-handled"),
}

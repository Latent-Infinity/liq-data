"""Simple rate limiter to enforce provider policies."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional


class RateLimiter:
    """Token-bucket-like limiter using sliding window."""

    def __init__(self, requests_per_minute: Optional[int] = None, burst: Optional[int] = None) -> None:
        self.requests_per_minute = requests_per_minute
        self.burst = burst or requests_per_minute
        self._events: deque[datetime] = deque()

    def acquire(self) -> None:
        """Block until within rate limits."""
        if not self.requests_per_minute:
            return
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=1)
        self._evict(now, window)
        if self.burst and len(self._events) >= self.burst:
            sleep_for = (self._events[0] + window - now).total_seconds()
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = datetime.now(timezone.utc)
            self._evict(now, window)
        self._events.append(now)

    def _evict(self, now: datetime, window: timedelta) -> None:
        while self._events and now - self._events[0] > window:
            self._events.popleft()

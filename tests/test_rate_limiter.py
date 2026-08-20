import concurrent.futures
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta

from liq.data.policies import POLICIES
from liq.data.rate_limiter import RateLimiter


def test_databento_policy_is_proactively_paced() -> None:
    policy = POLICIES["databento"]
    assert policy.requests_per_minute == 30
    assert policy.min_interval_seconds == 2.0


def test_rate_limiter_allows_without_policy() -> None:
    limiter = RateLimiter(requests_per_minute=None)
    limiter.acquire()  # should not block


def test_rate_limiter_blocks_when_burst_exceeded(monkeypatch) -> None:
    limiter = RateLimiter(requests_per_minute=2, burst=2)

    fake_now = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def fake_datetime_now(_tz):
        return fake_now

    monkeypatch.setattr(
        "liq.data.rate_limiter.datetime", type("dt", (), {"now": staticmethod(fake_datetime_now)})
    )

    sleep_called = []

    def fake_sleep(seconds: float) -> None:
        sleep_called.append(seconds)

    monkeypatch.setattr(
        "liq.data.rate_limiter.time", type("t", (), {"sleep": staticmethod(fake_sleep)})
    )

    limiter.acquire()
    limiter.acquire()
    # third call should trigger sleep
    limiter.acquire()
    assert sleep_called, "Expected rate limiter to sleep when burst exceeded"


def test_rate_limiter_enforces_min_interval(monkeypatch) -> None:
    limiter = RateLimiter(requests_per_minute=30, burst=30, min_interval_seconds=2.0)

    now = {"value": datetime(2024, 1, 1, 0, 0, tzinfo=UTC)}

    def fake_datetime_now(_tz):
        return now["value"]

    monkeypatch.setattr(
        "liq.data.rate_limiter.datetime", type("dt", (), {"now": staticmethod(fake_datetime_now)})
    )

    sleep_called = []

    def fake_sleep(seconds: float) -> None:
        sleep_called.append(seconds)
        now["value"] = now["value"] + timedelta(seconds=seconds)

    monkeypatch.setattr(
        "liq.data.rate_limiter.time", type("t", (), {"sleep": staticmethod(fake_sleep)})
    )

    limiter.acquire()
    limiter.acquire()

    assert sleep_called == [2.0]


def test_rate_limiter_serializes_concurrent_acquisitions() -> None:
    class ProbeEvents(deque):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak_active = 0
            self.probe_lock = threading.Lock()

        def append(self, value) -> None:
            with self.probe_lock:
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
            time.sleep(0.01)
            super().append(value)
            with self.probe_lock:
                self.active -= 1

    limiter = RateLimiter(requests_per_minute=100, burst=100)
    events = ProbeEvents()
    limiter._events = events

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _index: limiter.acquire(), range(8)))

    assert events.peak_active == 1

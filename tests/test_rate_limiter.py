from datetime import UTC, datetime

from liq.data.rate_limiter import RateLimiter


def test_rate_limiter_allows_without_policy() -> None:
    limiter = RateLimiter(requests_per_minute=None)
    limiter.acquire()  # should not block


def test_rate_limiter_blocks_when_burst_exceeded(monkeypatch) -> None:
    limiter = RateLimiter(requests_per_minute=2, burst=2)

    fake_now = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)

    def fake_datetime_now(_tz):
        return fake_now

    monkeypatch.setattr("liq.data.rate_limiter.datetime", type("dt", (), {"now": staticmethod(fake_datetime_now)}))

    sleep_called = []

    def fake_sleep(seconds: float) -> None:
        sleep_called.append(seconds)

    monkeypatch.setattr("liq.data.rate_limiter.time", type("t", (), {"sleep": staticmethod(fake_sleep)}))

    limiter.acquire()
    limiter.acquire()
    # third call should trigger sleep
    limiter.acquire()
    assert sleep_called, "Expected rate limiter to sleep when burst exceeded"

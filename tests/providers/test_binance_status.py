from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from liq.data.providers import binance_status as bs


class _FakeResponse:
    def __init__(self, payload: Any):
        self.payload = payload
        self.raised = False

    def json(self) -> Any:
        return self.payload

    def raise_for_status(self) -> None:
        self.raised = True


class _FakeClient:
    def __init__(self, responses: dict[str, Any]):
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        self.calls.append((url, params))
        payload = self.responses[url]
        return _FakeResponse(payload)


def test_fetch_binance_system_status(monkeypatch):
    responses = {
        "https://api.binance.com/sapi/v1/system/status": {"status": 0, "msg": "ok"}
    }

    # inject responses into fake client construction
    monkeypatch.setattr(bs, "httpx", SimpleNamespace(Client=lambda **kwargs: _FakeClient(responses)))

    result = bs.fetch_binance_system_status()

    assert result == {"status": 0, "msg": "ok"}


def test_fetch_binance_announcements(monkeypatch):
    announcements = {
        "histories": [
            {
                "incidents": [
                    {
                        "title": "maintenance",
                        "updates": [
                            {"code": "started", "timestamp": 1_700_000_000_000},
                            {"code": "resolved", "timestamp": 1_700_000_900_000},
                        ],
                    }
                ]
            }
        ]
    }
    responses = {"https://status.binance.com/status/spot/history": announcements}

    monkeypatch.setattr(bs, "httpx", SimpleNamespace(Client=lambda **kwargs: _FakeClient(responses)))

    result = bs.fetch_binance_announcements(limit=10)

    assert result == announcements["histories"]


def test_fetch_binance_announcements_handles_non_json(monkeypatch):
    class _NonJsonResponse(_FakeResponse):
        def json(self):
            raise ValueError("not json")

    class _ClientWithHtml(_FakeClient):
        def get(self, url: str, params=None):
            self.calls.append((url, params))
            return _NonJsonResponse("<html>")

    monkeypatch.setattr(bs, "httpx", SimpleNamespace(Client=lambda **kwargs: _ClientWithHtml({})))

    result = bs.fetch_binance_announcements(limit=5)

    assert result == []


def test_maintenance_windows_from_announcements():
    start = datetime(2023, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2023, 1, 1, 1, 0, tzinfo=UTC)
    data = [
        {
            "incidents": [
                {
                    "title": "maint",
                    "updates": [
                        {"code": "started", "timestamp": start.timestamp() * 1000},
                        {"code": "resolved", "timestamp": end.timestamp() * 1000},
                    ],
                }
            ]
        }
    ]

    windows = bs.maintenance_windows_from_announcements(data)

    assert windows == [(start, end, "maint")]

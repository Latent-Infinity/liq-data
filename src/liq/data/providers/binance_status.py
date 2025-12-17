"""Binance status/maintenance fetcher."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx


def fetch_binance_system_status() -> dict[str, Any]:
    """Fetch Binance system status."""
    with httpx.Client(timeout=10, follow_redirects=True) as client:
        resp = client.get("https://api.binance.com/sapi/v1/system/status")
        resp.raise_for_status()
        return resp.json()


def fetch_binance_announcements(limit: int = 50) -> list[dict[str, Any]]:
    """Fetch recent Binance system announcements (maintenance/updates)."""
    with httpx.Client(timeout=10, follow_redirects=True) as client:
        resp = client.get(
            "https://status.binance.com/status/spot/history", params={"limit": limit}
        )
        resp.raise_for_status()
        # response may be html if not supported; guard
        try:
            data = resp.json()
        except Exception:
            return []
        return data.get("histories", [])


def maintenance_windows_from_announcements(announcements: list[dict[str, Any]]) -> list[tuple[datetime, datetime, str]]:
    """Extract maintenance windows from status histories."""
    windows: list[tuple[datetime, datetime, str]] = []
    for item in announcements:
        for incident in item.get("incidents", []):
            name = incident.get("title", "maintenance")
            updates = incident.get("updates", [])
            start = None
            end = None
            for upd in updates:
                ts = upd.get("timestamp")
                try:
                    ts_dt = datetime.fromtimestamp(ts / 1000, tz=UTC)
                except Exception:
                    continue
                if upd.get("code") == "started":
                    start = ts_dt
                if upd.get("code") == "resolved":
                    end = ts_dt
            if start and end:
                windows.append((start, end, name))
    return windows

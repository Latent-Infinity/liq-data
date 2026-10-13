"""Guarded bid/ask candle access through the public data service."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

from liq.data.exceptions import LockboxViolationError, ProviderError, UndeclaredReadError
from liq.data.service import DataService
from liq.data.settings import LiqDataSettings


@pytest.fixture
def service(tmp_path: Path) -> DataService:
    settings = LiqDataSettings.model_construct(
        data_root=tmp_path,
        oanda_api_key="test-token",
        oanda_account_id="test-account",
        oanda_environment="practice",
    )
    return DataService(settings=settings)


@pytest.mark.parametrize(
    ("day", "purpose", "error"),
    [
        (date(2024, 1, 3), None, UndeclaredReadError),
        (date(2026, 1, 3), "discovery", LockboxViolationError),
    ],
)
def test_refuses_undeclared_or_protected_read_before_provider(
    service: DataService,
    monkeypatch: pytest.MonkeyPatch,
    day,
    purpose,
    error,
) -> None:
    def unexpected_provider(name: str):
        raise AssertionError("provider resolved before guard refusal")

    monkeypatch.setattr(service, "_get_provider", unexpected_provider)
    with pytest.raises(error):
        service.fetch_bid_ask_bars(
            "oanda",
            "EUR_USD",
            day,
            day,
            purpose=purpose,
            arm_id="bid_ask_service_tests",
            asset_class="forex",
        )


@respx.mock
def test_guard_logs_before_bid_ask_transport_and_no_storage_write(
    service: DataService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        entries = service.lockbox_guard.usage_log_path.read_text().splitlines()
        assert len(entries) == 1
        entry = json.loads(entries[0])
        assert entry["dataset"] == "oanda_fx"
        assert entry["purpose"] == "discovery"
        assert request.url.params["price"] == "BA"
        return httpx.Response(200, json={"candles": []})

    def unexpected_write(*args, **kwargs):
        raise AssertionError("bid/ask candles must not enter midpoint storage")

    monkeypatch.setattr(service.store, "write", unexpected_write)
    route = respx.get("https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles").mock(
        side_effect=response
    )
    frame = service.fetch_bid_ask_bars(
        "oanda",
        "EUR_USD",
        date(2024, 1, 3),
        date(2024, 1, 3),
        timeframe="1m",
        purpose="discovery",
        arm_id="bid_ask_service_tests",
        asset_class="forex",
    )
    assert route.call_count == 1
    assert frame.is_empty()
    assert "bid_close" in frame.columns and "ask_close" in frame.columns


def test_unsupported_provider_has_explicit_capability_error(
    service: DataService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "_get_provider", lambda _name: SimpleNamespace())
    with pytest.raises(ProviderError, match="bid/ask candles"):
        service.fetch_bid_ask_bars(
            "oanda",
            "EUR_USD",
            date(2024, 1, 3),
            date(2024, 1, 3),
            purpose="discovery",
            arm_id="bid_ask_service_tests",
            asset_class="forex",
        )

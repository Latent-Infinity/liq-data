"""Actual source guard before provider resolution, with no broker connection."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError as PayloadValidationError

from liq.data.exceptions import LockboxViolationError, UndeclaredReadError
from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.protocols import MarketDataProvider
from liq.data.providers.coinbase import CoinbaseProvider
from liq.data.providers.oanda_account_probe import PROBE_ARM
from liq.data.providers.oanda_price_capture import PriceCapturePolicyRefused
from liq.data.providers.oanda_source_body import SourceBody
from liq.data.service import DataService
from liq.data.settings import LiqDataSettings


@pytest.fixture
def service(tmp_path: Path) -> DataService:
    return DataService(
        settings=LiqDataSettings.model_construct(
            oanda_api_key="dummy-token",
            oanda_account_id="dummy-account",
            oanda_environment="live",
            data_root=tmp_path,
        ),
        data_root=tmp_path,
    )


def no_provider_resolution(_self: DataService, _name: str) -> MarketDataProvider:
    raise AssertionError("Refusal must precede provider resolution")


@pytest.mark.parametrize("day", [date(2020, 1, 3), date(2024, 1, 3), date(2024, 7, 3)])
@respx.mock
def test_actual_guard_is_durable_before_empty_envelope_transport(
    service: DataService, day: date
) -> None:
    # Given a fresh lazy guard and a real DataService/provider path.
    assert service._lockbox_guard is None

    def intercept(request: httpx.Request) -> httpx.Response:
        # This real ledger record must already exist before any transport.
        entries = service.lockbox_guard.usage_log_path.read_text().splitlines()
        assert len(entries) == 1
        entry = json.loads(entries[0])
        assert entry["dataset"] == "oanda_fx" and entry["purpose"] == "discovery"
        assert entry["start"] == entry["end"] == day.isoformat()
        assert entry["arm_id"] == PROBE_ARM and entry["final_portfolio_review"] is False
        assert request.url.host == "api-fxtrade.oanda.com"
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b'{"instrument":"EUR_USD","granularity":"M1","candles":[]}'),
        )

    route = respx.get(path__regex=r".*/candles$").mock(side_effect=intercept)
    # When the approved declaration reaches the guarded service facade.
    result = service.capture_oanda_source_prices(
        SourcePriceWindow(symbol="EUR_USD", day=day), purpose="discovery", arm_id=PROBE_ARM
    )
    # Then only quarantined structural bytes are returned and one GET occurred.
    assert isinstance(result, SourceBody) and route.call_count == 1
    assert service._lockbox_guard is not None


@pytest.mark.parametrize("purpose", [None, "", "validation", "paper_trading", "discovery "])
def test_wrong_purpose_has_zero_provider_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch, purpose: str | None
) -> None:
    monkeypatch.setattr(DataService, "_get_provider", no_provider_resolution)
    with pytest.raises((UndeclaredReadError, PriceCapturePolicyRefused)):
        service.capture_oanda_source_prices(
            SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3)),
            purpose=purpose,
            arm_id=PROBE_ARM,
        )
    assert service._lockbox_guard is None


@pytest.mark.parametrize("arm", ["", "different_arm", "idea_05b_london_fix"])
def test_wrong_arm_has_zero_provider_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    monkeypatch.setattr(DataService, "_get_provider", no_provider_resolution)
    with pytest.raises(PriceCapturePolicyRefused, match="DECLARATION_REFUSED"):
        service.capture_oanda_source_prices(
            SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3)),
            purpose="discovery",
            arm_id=arm,
        )
    assert service._lockbox_guard is None


@pytest.mark.parametrize(
    "scope",
    [
        SourcePriceWindow.model_construct(symbol="GBP_JPY", day=date(2024, 1, 3)),
        SourcePriceWindow.model_construct(symbol="EUR_USD", day=date(2025, 1, 3)),
        SourcePriceWindow.model_construct(symbol="EUR_USD", day=date(2026, 1, 3)),
    ],
)
def test_unsafe_constructed_case_is_rejected_before_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch, scope: SourcePriceWindow
) -> None:
    monkeypatch.setattr(DataService, "_get_provider", no_provider_resolution)
    with pytest.raises(PayloadValidationError):
        service.capture_oanda_source_prices(scope, purpose="discovery", arm_id=PROBE_ARM)
    assert service._lockbox_guard is None


@pytest.mark.parametrize("classification", [None, "sp500_adjusted"])
def test_missing_or_wrong_dataset_classification_stops_before_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch, classification: str | None
) -> None:
    def classify(*_args: str, **_kwargs: str | None) -> str | None:
        return classification

    monkeypatch.setattr(service.lockbox_guard, "classify_read", classify)
    monkeypatch.setattr(DataService, "_get_provider", no_provider_resolution)
    with pytest.raises(PriceCapturePolicyRefused, match="CLASSIFICATION_REFUSED"):
        service.capture_oanda_source_prices(
            SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3)),
            purpose="discovery",
            arm_id=PROBE_ARM,
        )


def test_guard_refusal_propagates_before_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: str | date | None, **_kwargs: str | bool) -> None:
        raise LockboxViolationError("Injected period refusal")

    monkeypatch.setattr(service.lockbox_guard, "assert_period_allowed", refuse)
    monkeypatch.setattr(DataService, "_get_provider", no_provider_resolution)
    with pytest.raises(LockboxViolationError):
        service.capture_oanda_source_prices(
            SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3)),
            purpose="discovery",
            arm_id=PROBE_ARM,
        )


def test_wrong_provider_capability_is_refused_after_guard_without_http(
    service: DataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(_self: DataService, _name: str) -> MarketDataProvider:
        return CoinbaseProvider()

    monkeypatch.setattr(DataService, "_get_provider", unsupported)
    with pytest.raises(PriceCapturePolicyRefused, match="PROVIDER_UNSUPPORTED"):
        service.capture_oanda_source_prices(
            SourcePriceWindow(symbol="EUR_USD", day=date(2024, 1, 3)),
            purpose="discovery",
            arm_id=PROBE_ARM,
        )
    entries = service.lockbox_guard.usage_log_path.read_text().splitlines()
    assert len(entries) == 1 and json.loads(entries[0])["dataset"] == "oanda_fx"

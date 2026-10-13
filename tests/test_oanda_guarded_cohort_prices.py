"""Cohort service declarations reach the actual guard before provider resolution."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from liq.data.exceptions import LockboxViolationError, UndeclaredReadError
from liq.data.fx_fix_cohort_window import CohortSourceWindow
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


def window(day: date = date(2023, 1, 2)) -> CohortSourceWindow:
    return CohortSourceWindow(
        scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=day
    )


def refuse_resolution(_self: DataService, _name: str) -> MarketDataProvider:
    pytest.fail("PROVIDER_RESOLUTION_BEFORE_REFUSAL")


@pytest.mark.parametrize("day", [date(2023, 1, 2), date(2023, 7, 3), date(2024, 3, 28)])
@respx.mock
def test_cohort_actual_guard_is_durable_before_empty_transport(
    service: DataService, day: date
) -> None:
    # Given a genuine service/provider path intercepted only at HTTP.
    def intercept(request: httpx.Request) -> httpx.Response:
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
    # When an exact discovery declaration enters the cohort facade.
    result = service.capture_oanda_cohort_prices(window(day), purpose="discovery", arm_id=PROBE_ARM)
    # Then the real guard preceded the single structurally quarantined response.
    assert isinstance(result, SourceBody) and route.call_count == 1


@pytest.mark.parametrize("purpose", [None, "", "validation", "paper_trading", "discovery "])
def test_cohort_wrong_purpose_refuses_before_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch, purpose: str | None
) -> None:
    # Given a provider-resolution sentinel and an invalid declaration.
    monkeypatch.setattr(DataService, "_get_provider", refuse_resolution)
    # When the closed service facade receives that declaration.
    with pytest.raises((UndeclaredReadError, PriceCapturePolicyRefused)):
        service.capture_oanda_cohort_prices(window(), purpose=purpose, arm_id=PROBE_ARM)
    # Then no research guard was initialized.
    assert service._lockbox_guard is None


def test_cohort_forged_scope_refuses_before_guard(
    service: DataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a forged unsupported date and a provider-resolution sentinel.
    monkeypatch.setattr(DataService, "_get_provider", refuse_resolution)
    forged = window().model_copy(update={"day": date(2025, 1, 2)})
    # When the service concretely reparses the cohort scope.
    with pytest.raises(PriceCapturePolicyRefused, match="WINDOW_REFUSED"):
        service.capture_oanda_cohort_prices(forged, purpose="discovery", arm_id=PROBE_ARM)
    # Then no guard/transport side effect occurred.
    assert service._lockbox_guard is None


def test_cohort_guard_refusal_propagates_before_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a real guard whose period boundary refuses the case.
    def refuse(*_args: str | date | None, **_kwargs: str | bool) -> None:
        raise LockboxViolationError("PERIOD_REFUSED")

    monkeypatch.setattr(service.lockbox_guard, "assert_period_allowed", refuse)
    monkeypatch.setattr(DataService, "_get_provider", refuse_resolution)
    # When the exact cohort declaration reaches that guard.
    with pytest.raises(LockboxViolationError, match="PERIOD_REFUSED"):
        service.capture_oanda_cohort_prices(window(), purpose="discovery", arm_id=PROBE_ARM)


@pytest.mark.parametrize("arm", ["", "different_arm", "idea_05b_london_fix"])
def test_cohort_wrong_arm_refuses_before_guard(
    service: DataService, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    # Given an undeclared arm and a resolution sentinel.
    monkeypatch.setattr(DataService, "_get_provider", refuse_resolution)
    # When ingress checks the exact arm.
    with pytest.raises(PriceCapturePolicyRefused, match="DECLARATION_REFUSED"):
        service.capture_oanda_cohort_prices(window(), purpose="discovery", arm_id=arm)
    # Then the actual guard was not initialized.
    assert service._lockbox_guard is None


@pytest.mark.parametrize("classification", [None, "sp500_adjusted"])
def test_cohort_wrong_classification_refuses_before_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch, classification: str | None
) -> None:
    # Given a missing or unrelated dataset classifier.
    def classify(*_args: str, **_kwargs: str | None) -> str | None:
        return classification

    monkeypatch.setattr(service.lockbox_guard, "classify_read", classify)
    monkeypatch.setattr(DataService, "_get_provider", refuse_resolution)
    # When the facade demands oanda_fx classification.
    with pytest.raises(PriceCapturePolicyRefused, match="CLASSIFICATION_REFUSED"):
        service.capture_oanda_cohort_prices(window(), purpose="discovery", arm_id=PROBE_ARM)


def test_cohort_unsupported_provider_refuses_after_durable_guard(
    service: DataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given an existing provider without the separate cohort capability.
    def unsupported(_self: DataService, _name: str) -> MarketDataProvider:
        return CoinbaseProvider()

    monkeypatch.setattr(DataService, "_get_provider", unsupported)
    # When the valid case reaches capability resolution after its actual guard.
    with pytest.raises(PriceCapturePolicyRefused, match="PROVIDER_UNSUPPORTED"):
        service.capture_oanda_cohort_prices(window(), purpose="discovery", arm_id=PROBE_ARM)
    # Then the guarded read remains recorded without any HTTP.
    entries = service.lockbox_guard.usage_log_path.read_text().splitlines()
    assert len(entries) == 1 and json.loads(entries[0])["dataset"] == "oanda_fx"

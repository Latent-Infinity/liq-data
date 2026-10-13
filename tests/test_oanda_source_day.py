"""Fixed-day source transport through the real guard, intercepted before HTTP."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from liq.data.exceptions import UndeclaredReadError, ValidationError
from liq.data.oanda_instrument_metadata import MATRIX_SYMBOLS
from liq.data.oanda_source_metadata import CandleDay
from liq.data.providers.oanda_account_probe import PROBE_ARM
from liq.data.service import DataService
from liq.data.settings import LiqDataSettings

DAYS = (
    date(2020, 1, 3),
    date(2024, 1, 3),
    date(2024, 1, 31),
    date(2024, 3, 15),
    date(2024, 7, 3),
    date(2024, 12, 30),
)


@respx.mock
def test_day_transport_error_is_not_retried(service: DataService) -> None:
    route = respx.get(path__regex=r".*/candles$").mock(
        side_effect=httpx.ConnectError("offline connection failure")
    )
    observation = service.probe_oanda_source_day(
        CandleDay("EUR_USD", date(2024, 1, 3)), purpose="discovery", arm_id=PROBE_ARM
    )
    assert observation.category == "TRANSPORT_ERROR"
    assert observation.http_status is None and observation.summary is None
    assert route.call_count == 1


@respx.mock
def test_day_schema_failure_retains_no_partial_summary(service: DataService) -> None:
    route = respx.get(path__regex=r".*/candles$").respond(200, json={})
    observation = service.probe_oanda_source_day(
        CandleDay("EUR_USD", date(2024, 1, 3)), purpose="discovery", arm_id=PROBE_ARM
    )
    assert observation.category == "SCHEMA_ERROR"
    assert observation.http_status == 200 and observation.summary is None
    assert route.call_count == 1


@pytest.fixture
def service(tmp_path: Path) -> DataService:
    return DataService(
        settings=LiqDataSettings.model_construct(
            data_root=tmp_path,
            oanda_api_key="offline-token",
            oanda_account_id="offline-account",
            oanda_environment="practice",
        )
    )


@pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
@pytest.mark.parametrize("day", DAYS)
@respx.mock
def test_exact_day_scope_is_guarded_before_one_account_get(
    service: DataService,
    symbol: str,
    day: date,
) -> None:
    def respond(incoming: httpx.Request) -> httpx.Response:
        entry = json.loads(service.lockbox_guard.usage_log_path.read_text().splitlines()[0])
        assert entry["start"] == entry["end"] == day.isoformat()
        assert entry["dataset"] == "oanda_fx" and entry["purpose"] == "discovery"
        assert entry["arm_id"] == PROBE_ARM and entry["final_portfolio_review"] is False
        assert dict(incoming.url.params) == {
            "from": f"{day.isoformat()}T00:00:00Z",
            "to": f"{day.isoformat()}T23:59:59.999999Z",
            "price": "BA",
            "granularity": "M1",
            "units": "1",
            "smooth": "false",
            "includeFirst": "true",
        }
        assert set(incoming.extensions["timeout"].values()) == {30.0}
        return httpx.Response(200, json={"instrument": symbol, "granularity": "M1", "candles": []})

    route = respx.get(
        f"https://api-fxpractice.oanda.com/v3/accounts/offline-account/instruments/{symbol}/candles"
    ).mock(side_effect=respond)

    observation = service.probe_oanda_source_day(
        CandleDay(symbol, day), purpose="discovery", arm_id=PROBE_ARM
    )

    assert observation.category == "OBSERVED" and observation.summary is not None
    assert len(observation.summary.difference.provider_absent) == 1440
    assert route.call_count == len(respx.calls) == 1


@pytest.mark.parametrize(
    "scope,purpose,arm",
    [
        (CandleDay("EUR_USD", date(2024, 1, 3)), None, PROBE_ARM),
        (CandleDay("EUR_USD", date(2024, 1, 3)), "validation", PROBE_ARM),
        (CandleDay("EUR_USD", date(2024, 1, 3)), "discovery", "other"),
        (CandleDay("EUR_USD", date(2025, 1, 3)), "discovery", PROBE_ARM),
        (CandleDay("EUR_USD", date(2024, 1, 4)), "discovery", PROBE_ARM),
        (CandleDay("EUR_GBP", date(2024, 1, 3)), "discovery", PROBE_ARM),
    ],
)
def test_changed_day_scope_refuses_before_provider_and_guard_log(
    service: DataService,
    monkeypatch: pytest.MonkeyPatch,
    scope: CandleDay,
    purpose: str | None,
    arm: str,
) -> None:
    monkeypatch.setattr(service, "_get_provider", lambda _: pytest.fail("provider resolved"))

    with pytest.raises((UndeclaredReadError, ValidationError)):
        service.probe_oanda_source_day(scope, purpose=purpose, arm_id=arm)

    assert not service.lockbox_guard.usage_log_path.exists()


@pytest.mark.parametrize("status", [302, 401, 429, 500])
@respx.mock
def test_failed_day_request_is_not_retried(service: DataService, status: int) -> None:
    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/offline-account/instruments/EUR_USD/candles"
    ).mock(
        return_value=httpx.Response(
            status, text="private", headers={"Location": "https://unapproved.invalid"}
        )
    )

    observation = service.probe_oanda_source_day(
        CandleDay("EUR_USD", date(2024, 1, 3)), purpose="discovery", arm_id=PROBE_ARM
    )

    assert observation.category == "HTTP_ERROR" and observation.http_status == status
    assert observation.summary is None and "private" not in repr(observation)
    assert route.call_count == len(respx.calls) == 1

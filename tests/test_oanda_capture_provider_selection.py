"""Capture-only resolution preserves the real guard and refuses generic paths."""

import json
from datetime import date
from pathlib import Path
from typing import Literal

import pytest

from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.protocols import MarketDataProvider
from liq.data.providers.oanda_account_probe import PROBE_ARM
from liq.data.providers.oanda_price_capture import PriceCapturePolicyRefused
from liq.data.service import DataService
from liq.data.settings import LiqDataSettings


class CaptureSelected(RuntimeError):
    def __init__(self) -> None:
        self.reason: Literal["CAPTURE_SELECTED"] = "CAPTURE_SELECTED"
        super().__init__(self.reason)


class CaptureOnlyService(DataService):
    def _get_provider(self, provider_name: str) -> MarketDataProvider:
        raise PriceCapturePolicyRefused("PROVIDER_UNSUPPORTED")

    def _get_oanda_capture_provider(self, window: SourcePriceWindow) -> MarketDataProvider:
        entries = self.lockbox_guard.usage_log_path.read_bytes().splitlines()
        assert len(entries) == 1
        entry = json.loads(entries[0])
        assert entry["dataset"] == "oanda_fx"
        assert entry["start"] == entry["end"] == window.day.isoformat()
        assert entry["purpose"] == "discovery" and entry["arm_id"] == PROBE_ARM
        raise CaptureSelected()


def test_guarded_capture_remains_available_with_generic_provider_paths_disabled(
    tmp_path: Path,
) -> None:
    service = CaptureOnlyService(
        settings=LiqDataSettings.model_construct(data_root=tmp_path),
        research_log_root=tmp_path,
    )
    with pytest.raises(CaptureSelected):
        service.capture_oanda_source_prices(
            SourcePriceWindow(symbol="EUR_USD", day=date(2024, 7, 3)),
            purpose="discovery",
            arm_id=PROBE_ARM,
        )


class RefusedIngressService(CaptureOnlyService):
    def _prepare_oanda_capture(self, window: SourcePriceWindow) -> None:
        assert window.symbol == "EUR_USD"
        raise CaptureSelected()


def test_capture_ingress_refusal_precedes_actual_guard_append(tmp_path: Path) -> None:
    service = RefusedIngressService(
        settings=LiqDataSettings.model_construct(data_root=tmp_path),
        research_log_root=tmp_path,
    )
    with pytest.raises(CaptureSelected):
        service.capture_oanda_source_prices(
            SourcePriceWindow(symbol="EUR_USD", day=date(2024, 7, 3)),
            purpose="discovery",
            arm_id=PROBE_ARM,
        )
    assert not service.lockbox_guard.usage_log_path.exists()


def test_default_capture_selector_preserves_keyword_and_generic_delegation(tmp_path: Path) -> None:
    service = CaptureOnlyService(settings=LiqDataSettings.model_construct(data_root=tmp_path))
    with pytest.raises(PriceCapturePolicyRefused) as refused:
        DataService._get_oanda_capture_provider(
            service, window=SourcePriceWindow(symbol="EUR_USD", day=date(2024, 7, 3))
        )
    assert refused.value.reason == "PROVIDER_UNSUPPORTED"

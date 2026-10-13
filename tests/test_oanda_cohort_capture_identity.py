"""A hook cannot change the case after its original ingress validation."""

from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.protocols import MarketDataProvider
from liq.data.providers.oanda_account_probe import PROBE_ARM
from liq.data.providers.oanda_price_capture import PriceCapturePolicyRefused
from liq.data.service import DataService
from liq.data.settings import LiqDataSettings


class MutatingIngress(DataService):
    def _prepare_oanda_cohort_capture(self, window: CohortSourceWindow) -> None:
        object.__setattr__(window, "day", date(2024, 3, 28))


class MutatingSelector(DataService):
    def _get_oanda_cohort_capture_provider(self, window: CohortSourceWindow) -> MarketDataProvider:
        object.__setattr__(window, "day", date(2024, 3, 28))
        return super()._get_oanda_cohort_capture_provider(window)


class SnapshotIdentityRefused(PermissionError):
    pass


class SnapshotIdentityObserved(PermissionError):
    pass


class RetainedIngress(DataService):
    """Mutable hook control retains the actual code-owned ingress object."""

    _selector_used = False

    def _prepare_oanda_cohort_capture(self, window: CohortSourceWindow) -> None:
        self._ingress = window

    def _get_oanda_cohort_capture_provider(self, window: CohortSourceWindow) -> MarketDataProvider:
        self._selector_used = True
        if window is not self._ingress:
            raise SnapshotIdentityRefused
        # This hook proves identity without constructing a provider or client.
        raise SnapshotIdentityObserved


class MutatingAfterGuard(RetainedIngress):
    """Preserve the existing guard hook signature; mutate only after its real append."""

    def _guard_research_read(
        self,
        provider: str,
        symbol: str,
        start: date | None,
        end: date | None,
        *,
        purpose: str | None,
        arm_id: str | None,
        final_portfolio_review: bool,
        asset_class: str | None = None,
        timeframe: str | None = None,
        data_kind: str = "bars",
    ) -> None:
        super()._guard_research_read(
            provider,
            symbol,
            start,
            end,
            purpose=purpose,
            arm_id=arm_id,
            final_portfolio_review=final_portfolio_review,
            asset_class=asset_class,
            timeframe=timeframe,
            data_kind=data_kind,
        )
        object.__setattr__(self._ingress, "day", date(2024, 3, 28))


def test_cohort_selector_retains_actual_ingress_identity(tmp_path: Path) -> None:
    # Given a genuine facade and an owner that latches its ingress object.
    service = RetainedIngress(
        settings=LiqDataSettings.model_construct(
            oanda_api_key="dummy-token",
            oanda_account_id="dummy-account",
            oanda_environment="live",
            data_root=tmp_path,
        )
    )
    window = CohortSourceWindow(
        scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=date(2023, 1, 2)
    )
    # When the real guard executes between the two hooks.
    with pytest.raises(SnapshotIdentityObserved):
        service.capture_oanda_cohort_prices(window, purpose="discovery", arm_id=PROBE_ARM)
    # Then an equal clone cannot replace the actual retained ingress.
    assert len(service.lockbox_guard.usage_log_path.read_bytes().splitlines()) == 1


def test_mutation_after_actual_guard_refuses_before_selector(tmp_path: Path) -> None:
    # Given a real guard append followed by mutation of its retained ingress.
    service = MutatingAfterGuard(
        settings=LiqDataSettings.model_construct(
            oanda_api_key="dummy-token",
            oanda_account_id="dummy-account",
            oanda_environment="live",
            data_root=tmp_path,
        )
    )
    window = CohortSourceWindow(
        scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=date(2023, 1, 2)
    )
    # When the genuine facade would select a provider using the changed object.
    with pytest.raises(PriceCapturePolicyRefused, match="WINDOW_REFUSED"):
        service.capture_oanda_cohort_prices(window, purpose="discovery", arm_id=PROBE_ARM)
    # Then the actual guard event exists but no provider hook is entered.
    assert not service._selector_used
    assert len(service.lockbox_guard.usage_log_path.read_bytes().splitlines()) == 1


@pytest.mark.parametrize("service_type", [MutatingIngress, MutatingSelector])
def test_cohort_owner_mutation_refuses_before_http(
    service_type: type[DataService], tmp_path: Path
) -> None:
    # Given a real service whose hook attempts to substitute another allowed day.
    service = service_type(
        settings=LiqDataSettings.model_construct(
            oanda_api_key="dummy-token",
            oanda_account_id="dummy-account",
            oanda_environment="live",
            data_root=tmp_path,
        ),
        data_root=tmp_path,
    )
    window = CohortSourceWindow(
        scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=date(2023, 1, 2)
    )
    with respx.mock(assert_all_called=False) as router:
        route = router.get(path__regex=r".*/candles$").mock(
            return_value=httpx.Response(
                200,
                stream=httpx.ByteStream(
                    b'{"instrument":"EUR_USD","granularity":"M1","candles":[]}'
                ),
            )
        )
        # When ingress or selector tries to change the validated case.
        with pytest.raises(PriceCapturePolicyRefused, match="WINDOW_REFUSED"):
            service.capture_oanda_cohort_prices(window, purpose="discovery", arm_id=PROBE_ARM)
        # Then no request leaves the fixed audited/declared identity.
        assert route.call_count == 0

"""The shared stream must not rely on callers going through public wrappers."""

from datetime import date

import httpx
import pytest

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_source_capture_contract import SourcePriceWindow
from liq.data.providers.oanda import OandaProvider
from liq.data.providers.oanda_price_capture import PriceCaptureFailure
from liq.data.providers.oanda_price_request import _SourcePriceRequest, _stream_source_prices


class AlternateLegacyWindow(SourcePriceWindow):
    """A different physical type is not the private stream's fixed legacy case."""


class AlternateCohortWindow(CohortSourceWindow):
    """A different physical type is not the private stream's fixed cohort case."""


@pytest.mark.parametrize(
    "window",
    [
        SourcePriceWindow.model_construct(symbol="GBP_JPY", day=date(2025, 1, 2)),
        SourcePriceWindow.model_construct(symbol="EUR_USD", day=date(2025, 1, 2)),
        CohortSourceWindow.model_construct(
            scope="london-fx-fix-development-source/v1", symbol="GBP_JPY", day=date(2025, 1, 2)
        ),
        CohortSourceWindow.model_construct(
            scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=date(2025, 1, 2)
        ),
        CohortSourceWindow.model_construct(symbol="EUR_USD", day=date(2023, 1, 2)),
        AlternateLegacyWindow(symbol="EUR_USD", day=date(2024, 1, 3)),
        AlternateCohortWindow(
            scope="london-fx-fix-development-source/v1", symbol="EUR_USD", day=date(2023, 1, 2)
        ),
    ],
)
def test_private_stream_forged_window_refuses_before_client(
    window: SourcePriceWindow | CohortSourceWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a directly constructed request without either public wrapper.
    source = OandaProvider(api_key="dummy-token", account_id="dummy-account", environment="live")
    request = _SourcePriceRequest(window)

    def unexpected_client() -> httpx.Client:
        pytest.fail("CLIENT_REACHED_BY_FORGED_PRIVATE_REQUEST")

    monkeypatch.setattr(source, "_get_client", unexpected_client)
    # When the private stream is called without a concrete public wrapper.
    result = _stream_source_prices(source, request)
    # Then no client resolves and only a safe refusal exists.
    assert result == PriceCaptureFailure("WINDOW_REFUSED", None, 0)

"""Offline source-summary contracts; no fabricated financial observations."""

from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx
import pytest
from test_oanda_bid_ask import PUBLISHED_CANDLE

import liq.data.oanda_source_metadata as metadata
from liq.data.exceptions import ValidationError
from liq.data.oanda_source_metadata import CandleDay, summarize_candle_day


def test_empty_day_reports_absence_without_repair() -> None:
    response = httpx.Response(
        200, json={"instrument": "EUR_USD", "granularity": "M1", "candles": []}
    )

    summary = summarize_candle_day(response, CandleDay("EUR_USD", date(2024, 1, 3)))

    assert summary.raw_count == summary.complete_count == summary.accepted_count == 0
    assert len(summary.difference.provider_absent) == 1440
    assert summary.difference.provider_absent[0] == datetime(2024, 1, 3, tzinfo=UTC)
    assert summary.difference.provider_absent[-1] == datetime(2024, 1, 3, 23, 59, tzinfo=UTC)
    assert summary.difference.raw_incomplete == summary.difference.adapter_dropped == ()


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_http_errors_never_publish_a_source_summary(status: int) -> None:
    response = httpx.Response(status, text="private provider response")

    with pytest.raises(ValidationError) as caught:
        summarize_candle_day(response, CandleDay("EUR_USD", date(2024, 1, 3)))

    assert "private provider response" not in str(caught.value)


@pytest.mark.parametrize("instrument,granularity", [("GBP_USD", "M1"), ("EUR_USD", "M5")])
def test_response_identity_must_match_declared_minute_day(
    instrument: str, granularity: str
) -> None:
    response = httpx.Response(
        200, json={"instrument": instrument, "granularity": granularity, "candles": []}
    )

    with pytest.raises(ValidationError):
        summarize_candle_day(response, CandleDay("EUR_USD", date(2024, 1, 3)))


def test_missing_candles_is_not_reported_as_an_empty_day() -> None:
    response = httpx.Response(200, json={"instrument": "EUR_USD", "granularity": "M1"})

    with pytest.raises(ValidationError):
        summarize_candle_day(response, CandleDay("EUR_USD", date(2024, 1, 3)))


@pytest.mark.parametrize("complete", [True, False])
def test_published_candle_projects_completion_without_retaining_prices(complete: bool) -> None:
    # Unchanged prices/time from the provenance-cited third-party M5 example.
    # M1 envelope and completion mutation are offline API-shape metadata, not
    # an authenticated M1 capture or evidence for any proposed matrix date.
    candle = deepcopy(PUBLISHED_CANDLE)
    candle["complete"] = complete
    response = httpx.Response(
        200, json={"instrument": "EUR_USD", "granularity": "M1", "candles": [candle]}
    )

    summary = summarize_candle_day(response, CandleDay("EUR_USD", date(2016, 1, 3)))

    timestamp = datetime(2016, 1, 3, 22, tzinfo=UTC)
    assert summary.raw_count == 1
    assert summary.complete_count == summary.accepted_count == int(complete)
    assert len(summary.difference.provider_absent) == 1439
    assert summary.accepted_timestamps == ((timestamp,) if complete else ())
    assert summary.difference.raw_incomplete == (() if complete else (timestamp,))
    assert summary.difference.adapter_dropped == ()
    assert "1.08699" not in repr(summary)


def test_duplicate_published_timestamp_is_rejected() -> None:
    response = httpx.Response(
        200,
        json={
            "instrument": "EUR_USD",
            "granularity": "M1",
            "candles": [PUBLISHED_CANDLE, PUBLISHED_CANDLE],
        },
    )

    with pytest.raises(ValidationError):
        summarize_candle_day(response, CandleDay("EUR_USD", date(2016, 1, 3)))


def test_published_timestamp_outside_declared_day_is_rejected() -> None:
    response = httpx.Response(
        200, json={"instrument": "EUR_USD", "granularity": "M1", "candles": [PUBLISHED_CANDLE]}
    )

    with pytest.raises(ValidationError):
        summarize_candle_day(response, CandleDay("EUR_USD", date(2024, 1, 3)))


def test_unordered_invalid_metadata_mutation_is_rejected() -> None:
    # Negative metadata mutation only; these are not two observed market bars.
    earlier = deepcopy(PUBLISHED_CANDLE)
    earlier["time"] = "2016-01-03T21:00:00.000000000Z"
    response = httpx.Response(
        200,
        json={"instrument": "EUR_USD", "granularity": "M1", "candles": [PUBLISHED_CANDLE, earlier]},
    )

    with pytest.raises(ValidationError, match="unordered"):
        summarize_candle_day(response, CandleDay("EUR_USD", date(2016, 1, 3)))


def test_decoder_identity_corruption_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fault injection swaps existing real price into an identity slot; it must
    # never become timestamp or financial evidence in the metadata projection.
    decoder = metadata._bid_ask_rows

    def corrupt_identity(
        payload: metadata._DayEnvelope, start: datetime, end: datetime
    ) -> list[dict[str, datetime | Decimal]]:
        rows = decoder(payload, start, end)
        rows[0]["timestamp"] = rows[0]["bid_close"]
        return rows

    monkeypatch.setattr(metadata, "_bid_ask_rows", corrupt_identity)
    response = httpx.Response(
        200, json={"instrument": "EUR_USD", "granularity": "M1", "candles": [PUBLISHED_CANDLE]}
    )

    with pytest.raises(ValidationError, match="non-timestamp"):
        summarize_candle_day(response, CandleDay("EUR_USD", date(2016, 1, 3)))

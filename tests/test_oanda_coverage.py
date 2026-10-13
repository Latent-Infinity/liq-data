"""Offline boundary tests for a guarded OANDA timestamp diagnostic."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from liq.data.oanda_coverage import (
    CandleMetadata,
    RequestBoundaryError,
    RequestScope,
    RequestTelemetry,
    capture_coverage,
    compare_timestamps,
    extract_candle_metadata,
    require_in_scope_request,
)


@pytest.mark.parametrize("timing", ["normal", "late_completion", "slow_get"])
def test_capture_coverage_uses_one_guarded_get_and_same_adapter_summary(
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
) -> None:
    import polars as pl

    import liq.data.service as data_service

    url = (
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        "?from=2024-01-03T00%3A00%3A00Z&to=2024-01-03T00%3A01%3A00Z"
        "&granularity=M1&price=M"
    )

    def offline_get(client: httpx.Client, request_url: str, **kwargs: object) -> httpx.Response:
        assert kwargs["follow_redirects"] is False
        assert isinstance(kwargs["timeout"], (int, float))
        assert kwargs["timeout"] <= 30
        return httpx.Response(
            200,
            json={"candles": [{"time": DAY.isoformat(), "complete": True}]},
            request=httpx.Request("GET", request_url),
        )

    class OfflineService:
        def probe_bar_coverage(self) -> data_service.BarCoverageObservation:
            httpx.Client().get(url)
            return data_service._summarize_bar_coverage(
                pl.DataFrame({"timestamp": [DAY]}),
                provider="oanda",
                symbol="EUR_USD",
                timeframe="1m",
                start=DAY.date(),
                end=DAY.date(),
            )

    monkeypatch.setattr(httpx.Client, "get", offline_get)
    if timing != "normal":
        import liq.data.oanda_coverage as diagnostic

        ticks = iter((0.0, 0.0, 31.0, 31.0) if timing == "slow_get" else (0.0, 0.0, 1.0, 61.0))
        monkeypatch.setattr(diagnostic.time, "monotonic", lambda: next(ticks))
        message = "request budget" if timing == "slow_get" else "deadline exhausted"
        with pytest.raises(RequestBoundaryError, match=message):
            capture_coverage(
                OfflineService().probe_bar_coverage,
                RequestScope(
                    host="api-fxpractice.oanda.com",
                    start=DAY,
                    end=DAY + timedelta(minutes=1),
                ),
            )
        return
    result = capture_coverage(
        OfflineService().probe_bar_coverage,
        RequestScope(
            host="api-fxpractice.oanda.com",
            start=DAY,
            end=DAY + timedelta(minutes=1),
        ),
    )
    assert result.http_get_count == 1
    assert result.difference.provider_absent == (DAY + timedelta(minutes=1),)
    assert result.difference.raw_incomplete == ()
    assert result.difference.adapter_dropped == ()


def test_capture_coverage_refuses_second_get_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import liq.data.service as data_service

    url = (
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        "?from=2024-01-03T00%3A00%3A00Z&to=2024-01-03T00%3A01%3A00Z"
        "&granularity=M1&price=M"
    )
    sent = 0

    def offline_get(client: httpx.Client, request_url: str, **kwargs: object) -> httpx.Response:
        nonlocal sent
        sent += 1
        return httpx.Response(
            200,
            json={"candles": [{"time": DAY.isoformat(), "complete": True}]},
            request=httpx.Request("GET", request_url),
        )

    def probe() -> data_service.BarCoverageObservation:
        httpx.Client().get(url)
        httpx.Client().get(url)
        raise AssertionError("second GET should be refused")

    monkeypatch.setattr(httpx.Client, "get", offline_get)
    with pytest.raises(RequestBoundaryError, match="unapproved OANDA request"):
        capture_coverage(
            probe,
            RequestScope(
                host="api-fxpractice.oanda.com",
                start=DAY,
                end=DAY + timedelta(minutes=1),
            ),
        )
    assert sent == 1


DAY = datetime(2024, 1, 3, tzinfo=UTC)
END = DAY + timedelta(days=1) - timedelta(microseconds=1)


def test_comparison_separates_provider_omission_incomplete_and_adapter_drop() -> None:
    # Given raw timestamp/flag metadata and the timestamps accepted by the adapter.
    raw = (
        CandleMetadata(time=DAY, complete=True),
        CandleMetadata(time=DAY + timedelta(minutes=2), complete=True),
        CandleMetadata(time=DAY + timedelta(minutes=3), complete=False),
    )
    accepted = (DAY,)

    # When the diagnostic compares the same four-minute window.
    result = compare_timestamps(raw, accepted, DAY, DAY + timedelta(minutes=3))

    # Then each distinct cause remains visible without fabricating a candle.
    assert result.provider_absent == (DAY + timedelta(minutes=1),)
    assert result.raw_incomplete == (DAY + timedelta(minutes=3),)
    assert result.adapter_dropped == (DAY + timedelta(minutes=2),)


def test_comparison_refuses_accepted_timestamp_missing_from_raw_response() -> None:
    # Given an adapter timestamp not present in raw metadata.
    raw = (CandleMetadata(time=DAY, complete=True),)

    # When the diagnostic compares raw and accepted timestamps.
    with pytest.raises(RequestBoundaryError):
        compare_timestamps(raw, (DAY + timedelta(minutes=1),), DAY, DAY + timedelta(minutes=1))


def test_comparison_refuses_naive_accepted_timestamp() -> None:
    # Given an accepted timestamp with no timezone declaration.
    raw = (CandleMetadata(time=DAY, complete=True),)

    # When it is compared with timezone-aware provider metadata.
    with pytest.raises(RequestBoundaryError):
        compare_timestamps(raw, (DAY.replace(tzinfo=None),), DAY, DAY + timedelta(minutes=1))


@pytest.mark.parametrize(
    "url",
    [
        "https://api-fxtrade.oanda.com/v3/instruments/EUR_USD/candles?from=2024-01-03T00%3A00%3A00.000000Z&to=2024-01-03T23%3A59%3A59.999999Z&granularity=M1&price=M",
        "https://api-fxpractice.oanda.com/v3/instruments/USD_JPY/candles?from=2024-01-03T00%3A00%3A00.000000Z&to=2024-01-03T23%3A59%3A59.999999Z&granularity=M1&price=M",
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles?from=2024-01-03T00%3A00%3A00.000000Z&to=2024-01-03T23%3A59%3A59.999999Z&granularity=M1&price=M&count=5000",
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles?from=2024-01-04T00%3A00%3A00.000000Z&to=2024-01-04T23%3A59%3A59.999999Z&granularity=M1&price=M",
    ],
)
def test_request_boundary_refuses_wrong_host_symbol_query_or_day(url: str) -> None:
    # Given the prospectively reviewed one-day practice request scope.
    scope = RequestScope(host="api-fxpractice.oanda.com", start=DAY, end=END)

    # When an endpoint or query differs from that scope.
    with pytest.raises(RequestBoundaryError):
        require_in_scope_request(url, scope, request_count=0)


def test_request_boundary_allows_exact_first_get_and_refuses_second() -> None:
    # Given the reviewed host, symbol, midpoint granularity, and UTC bounds.
    scope = RequestScope(host="api-fxpractice.oanda.com", start=DAY, end=END)
    url = (
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        "?from=2024-01-03T00%3A00%3A00.000000Z"
        "&to=2024-01-03T23%3A59%3A59.999999Z&granularity=M1&price=M"
    )

    # When the URL is checked before any HTTP request.
    require_in_scope_request(url, scope, request_count=0)

    # Then the same process cannot spend a second GET on this one-day window.
    with pytest.raises(RequestBoundaryError):
        require_in_scope_request(url, scope, request_count=1)


def test_response_projection_keeps_only_timestamp_and_complete_flag() -> None:
    response = httpx.Response(
        200,
        json={
            "candles": [
                {"time": "2024-01-03T00:00:00.000000000Z", "complete": True, "unused": "discard"},
            ],
        },
    )

    result = extract_candle_metadata(response)

    assert result == (CandleMetadata(time=DAY, complete=True),)


def test_response_projection_refuses_missing_complete_flag() -> None:
    response = httpx.Response(200, json={"candles": [{"time": "2024-01-03T00:00:00Z"}]})

    with pytest.raises(RequestBoundaryError):
        extract_candle_metadata(response)


def test_response_projection_refuses_http_error() -> None:
    response = httpx.Response(429, request=httpx.Request("GET", "https://example.test"))
    with pytest.raises(RequestBoundaryError, match="not successful"):
        extract_candle_metadata(response)


@pytest.mark.parametrize(
    "timestamp",
    ["2024-01-03T00:00:00.000000001Z", "2024-01-03T00:00:01Z", 123],
)
def test_response_projection_refuses_subminute_or_nontext_time(timestamp: object) -> None:
    response = httpx.Response(
        200,
        json={"candles": [{"time": timestamp, "complete": True}]},
        request=httpx.Request("GET", "https://example.test"),
    )
    with pytest.raises(RequestBoundaryError):
        extract_candle_metadata(response)


def test_response_projection_accepts_zero_filled_nanoseconds() -> None:
    response = httpx.Response(
        200,
        json={"candles": [{"time": "2024-01-03T00:00:00.000000000Z", "complete": True}]},
        request=httpx.Request("GET", "https://example.test"),
    )
    assert extract_candle_metadata(response) == (CandleMetadata(DAY, True),)


def test_request_boundary_refuses_malformed_utc_bounds() -> None:
    url = (
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        "?from=not-a-date&to=2024-01-03T00%3A01%3A00Z&granularity=M1&price=M"
    )
    with pytest.raises(RequestBoundaryError, match="invalid UTC request bounds"):
        require_in_scope_request(
            url,
            RequestScope(
                host="api-fxpractice.oanda.com",
                start=DAY,
                end=DAY + timedelta(minutes=1),
            ),
            request_count=0,
        )


def test_comparison_refuses_invalid_utc_window() -> None:
    with pytest.raises(RequestBoundaryError, match="invalid UTC comparison window"):
        compare_timestamps((), (), DAY, DAY)


def test_capture_retains_request_telemetry_after_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import liq.data.service as data_service

    url = (
        "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
        "?from=2024-01-03T00%3A00%3A00Z&to=2024-01-03T00%3A01%3A00Z"
        "&granularity=M1&price=M"
    )

    def offline_get(client: httpx.Client, request_url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("offline transport failure")

    def probe() -> data_service.BarCoverageObservation:
        httpx.Client().get(url)
        raise AssertionError("transport should fail")

    monkeypatch.setattr(httpx.Client, "get", offline_get)
    telemetry = RequestTelemetry()
    with pytest.raises(httpx.ConnectError):
        capture_coverage(
            probe,
            RequestScope(
                host="api-fxpractice.oanda.com",
                start=DAY,
                end=DAY + timedelta(minutes=1),
            ),
            telemetry=telemetry,
        )
    assert telemetry.get_count == 1
    assert telemetry.request_started_utc is not None
    assert telemetry.request_ended_utc is not None
    assert telemetry.request_from_utc == DAY.isoformat()
    assert telemetry.http_status is None

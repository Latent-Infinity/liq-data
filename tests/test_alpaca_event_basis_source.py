"""Nonfinancial controls for the finite provider-adjusted arithmetic source."""

import json
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from announcement_test_support import isolated_announcement_http as isolated_announcement_http
from pydantic import ValidationError

from liq.data.alpaca_announcement_source import (
    ANNOUNCEMENT_DATE_PAIRS,
    AlpacaAnnouncementDailyRequest,
)
from liq.data.alpaca_event_basis_source import AlpacaEventBasisDailyRequest
from liq.data.alpaca_source import AlpacaCaptureRefused
from liq.data.exceptions import LockboxViolationError
from liq.data.lockbox import (
    CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1,
    CLASSICAL_RANKER_EVENT_BASIS_FIXTURE_LEDGER_V1,
    resolve_dataset,
)
from liq.data.providers.alpaca import AlpacaProvider
from liq.data.service import DataService


class OpaqueBody(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"{}"


def request() -> AlpacaEventBasisDailyRequest:
    return AlpacaEventBasisDailyRequest(
        symbol="SPY", accession="0001637459-23-000005", adjustment="all"
    )


@pytest.mark.parametrize("accession", tuple(ANNOUNCEMENT_DATE_PAIRS))
@pytest.mark.parametrize("symbol", ["KHC", "SPY"])
@pytest.mark.parametrize("adjustment", ["raw", "all"])
def test_exact_sixteen_wire_slots(accession: str, symbol: str, adjustment: str) -> None:
    candidate = AlpacaEventBasisDailyRequest.model_validate(
        {"symbol": symbol, "accession": accession, "adjustment": adjustment}
    )
    seen: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        seen.append(wire)
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=OpaqueBody()
        )

    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    provider._source_transport = httpx.MockTransport(respond)
    assert provider.fetch_source_page(candidate, page_token=None) == b"{}"
    assert len(seen) == 1
    query = dict(seen[0].url.params)
    assert query == {
        "symbols": symbol,
        "start": candidate.start.isoformat().replace("+00:00", "Z"),
        "end": candidate.end.isoformat().replace("+00:00", "Z"),
        "timeframe": "1Day",
        "adjustment": adjustment,
        "asof": "2023-02-15",
        "feed": "sip",
        "currency": "USD",
        "limit": "4",
        "sort": "asc",
    }
    assert candidate.start.date() == ANNOUNCEMENT_DATE_PAIRS[candidate.accession][0]
    assert candidate.end.date() == ANNOUNCEMENT_DATE_PAIRS[candidate.accession][1]
    assert candidate.start.hour == (5 if accession.endswith("000005") else 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("symbol", "META"),
        ("accession", "other"),
        ("adjustment", "split"),
        ("feed", "iex"),
        ("asof", date(2024, 1, 1)),
        ("max_pages", 2),
        ("max_response_bytes", 262144),
        ("extra", True),
    ],
)
def test_scope_cannot_be_widened(field: str, value: object) -> None:
    values = request().model_dump()
    values[field] = value
    with pytest.raises((ValidationError, AlpacaCaptureRefused)):
        AlpacaEventBasisDailyRequest.model_validate(values)


def test_mutation_subclass_and_copy_refuse_before_http() -> None:
    original = request()
    with pytest.raises(ValidationError):
        original.adjustment = "raw"

    class Altered(AlpacaEventBasisDailyRequest):
        pass

    candidates = [Altered(**original.model_dump()), original.model_copy(update={"feed": "iex"})]
    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    for candidate in candidates:
        with pytest.raises((ValidationError, AlpacaCaptureRefused)):
            provider.fetch_source_page(candidate, page_token=None)


@pytest.mark.parametrize("status", [302, 401, 403, 429, 503])
def test_no_retry_or_redirect(status: int) -> None:
    seen: list[httpx.Request] = []

    def respond(wire: httpx.Request) -> httpx.Response:
        seen.append(wire)
        return httpx.Response(status, headers={"location": "https://example.invalid"})

    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    provider._source_transport = httpx.MockTransport(respond)
    with pytest.raises(AlpacaCaptureRefused):
        provider.fetch_source_page(request(), page_token=None)
    assert len(seen) == 1
    with pytest.raises(AlpacaCaptureRefused, match="continuation"):
        provider.fetch_source_page(request(), page_token="next")
    assert len(seen) == 1


def test_separate_guard_and_usage_kind(tmp_path: Path) -> None:
    assert (
        resolve_dataset("alpaca", "SPY", data_kind="ranker_event_basis_all_fixture")
        == "alpaca_ranker_event_basis_all_fixture"
    )
    assert (
        resolve_dataset("alpaca", "SPY", data_kind="ranker_event_basis_raw_fixture")
        == "alpaca_ranker_event_basis_raw_fixture"
    )
    assert resolve_dataset("other", "SPY", data_kind="ranker_event_basis_all_fixture") is None
    for ledger in (
        CLASSICAL_RANKER_EVENT_BASIS_FIXTURE_LEDGER_V1,
        CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1,
    ):
        service = DataService(data_root=tmp_path, lockbox_ledger=ledger)
        with patch.object(service, "_get_provider") as provider:
            if ledger == CLASSICAL_RANKER_ANNOUNCEMENT_FIXTURE_LEDGER_V1:
                with pytest.raises(LockboxViolationError):
                    service.fetch_alpaca_source_page(
                        request(),
                        page_token=None,
                        purpose="characterization",
                        arm_id="classical_cross_sectional_ranker",
                    )
                provider.assert_not_called()
            else:
                with (
                    patch.object(service, "_guard_research_read") as guard,
                    pytest.raises(ValueError, match="concrete"),
                ):
                    service.fetch_alpaca_source_page(
                        request(),
                        page_token=None,
                        purpose="characterization",
                        arm_id="classical_cross_sectional_ranker",
                    )
                assert guard.call_args.kwargs["data_kind"] == "ranker_event_basis_all_fixture"
                provider.assert_called_once()


def test_old_raw_contract_is_unchanged() -> None:
    with pytest.raises(ValidationError):
        AlpacaAnnouncementDailyRequest.model_validate(
            {"symbol": "SPY", "accession": "0001637459-23-000005", "adjustment": "all"}
        )


@pytest.mark.parametrize("adjustment", ["raw", "all"])
def test_actual_guard_appends_distinct_usage_before_transport(
    tmp_path: Path, adjustment: str
) -> None:
    candidate = AlpacaEventBasisDailyRequest.model_validate(
        {"symbol": "SPY", "accession": "0001637459-23-000005", "adjustment": adjustment}
    )
    service = DataService(
        data_root=tmp_path / "data",
        research_log_root=tmp_path,
        lockbox_ledger=CLASSICAL_RANKER_EVENT_BASIS_FIXTURE_LEDGER_V1,
    )
    log = tmp_path / "lockbox_usage_log.jsonl"
    provider = AlpacaProvider("CONTROL_KEY", "CONTROL_SECRET")
    attempts: list[bool] = []

    def respond(_wire: httpx.Request) -> httpx.Response:
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["dataset"] == f"alpaca_ranker_event_basis_{adjustment}_fixture"
        assert rows[0]["start"] == "2023-02-14"
        assert rows[0]["end"] == "2023-02-15"
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=OpaqueBody()
        )

    provider._source_transport = httpx.MockTransport(respond)
    with patch.object(service, "_get_provider", return_value=provider):
        assert (
            service.fetch_alpaca_source_page(
                candidate,
                page_token=None,
                purpose="characterization",
                arm_id="classical_cross_sectional_ranker",
                on_http_attempt=lambda: attempts.append(True),
            )
            == b"{}"
        )
    assert attempts == [True]

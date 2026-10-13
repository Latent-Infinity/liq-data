"""Nonfinancial symbology/authority controls, not historical market-data fixtures."""

import hashlib
import json
from dataclasses import replace
from unittest.mock import Mock

import httpx
import pytest

from liq.data.lockbox import INTRADAY_CAMPAIGN_LEDGER_V1, LockboxGuard
from liq.data.providers import databento_symbology as symbology
from liq.data.providers._databento_transport import BoundedResponse


def _document():
    return {
        "result": {"CONTROL": [{"d0": "2018-05-01", "d1": "2018-05-02", "s": "771"}]},
        "symbols": ["CONTROL"],
        "stype_in": "raw_symbol",
        "stype_out": "instrument_id",
        "start_date": "2018-05-01",
        "end_date": "2018-05-02",
        "partial": [],
        "not_found": [],
        "message": "OK",
        "status": 0,
    }


def _raw(document=None, *, body=None):
    return BoundedResponse(
        200,
        json.dumps(_document() if document is None else document).encode()
        if body is None
        else body,
        "application/json",
        "",
    )


def test_descriptor_retains_interval_and_does_not_claim_universe_or_pit():
    snapshot = symbology.parse_symbology(_raw(), symbology.SymbologyRequest())
    assert snapshot.mappings[0].raw_symbol == "CONTROL"
    assert snapshot.mappings[0].intervals[0].s == "771"
    assert snapshot.mappings[0].intervals[0].d1 == "2018-05-02"
    assert snapshot.completeness_claim == "NOT_ESTABLISHED"
    assert snapshot.pit_vintage_claim == "NOT_ESTABLISHED"


def test_partial_and_not_found_are_preserved_without_imputation():
    document = _document()
    document.update(partial=["CONTROL"], not_found=["UNRESOLVED_CONTROL"])
    document["symbols"] = ["CONTROL", "UNRESOLVED_CONTROL"]
    snapshot = symbology.parse_symbology(_raw(document), symbology.SymbologyRequest())
    assert snapshot.partial == ("CONTROL",)
    assert snapshot.not_found == ("UNRESOLVED_CONTROL",)


@pytest.mark.parametrize(
    "change", ["date", "interval", "id", "duplicates", "status", "stype", "classifications"]
)
def test_bad_descriptor_refuses_without_repair(change):
    document = _document()
    if change == "date":
        document["start_date"] = "2018-05-02"
    elif change == "interval":
        document["result"]["CONTROL"][0]["d0"] = "2018-04-30"
    elif change == "id":
        document["result"]["CONTROL"][0]["s"] = None
    elif change == "duplicates":
        document["result"]["CONTROL"] *= 2
    elif change == "status":
        document["status"] = True
    elif change == "stype":
        document["stype_out"] = "raw_symbol"
    else:
        document["partial"] = ["CONTROL", "CONTROL"]
    with pytest.raises(symbology.SymbologyRefused):
        symbology.parse_symbology(_raw(document), symbology.SymbologyRequest())


@pytest.mark.parametrize(
    "body", [b'{"result":{},"result":{}}', b'{"status":NaN}', b"[]", b"not-json"]
)
def test_invalid_wire_descriptor_refuses(body):
    with pytest.raises(symbology.SymbologyRefused):
        symbology.parse_symbology(_raw(body=body), symbology.SymbologyRequest())


@pytest.mark.parametrize("change", ["dataset", "symbols", "start_date", "end_date"])
def test_request_scope_is_exact(change):
    values = {
        "dataset": "EQUS.MINI",
        "symbols": "KHC,SPY",
        "start_date": "2018-05-02",
        "end_date": "2018-05-03",
    }
    with pytest.raises(symbology.SymbologyRefused, match="request_scope"):
        replace(symbology.SymbologyRequest(), **{change: values[change]}).parameters()


def _fetch(guard, **kwargs):
    return symbology.fetch_symbology(
        symbology.SymbologyRequest(),
        guard=guard,
        purpose="characterization",
        arm_id=symbology.SYMBOLOGY_ARM,
        expected_api_key_sha256=hashlib.sha256(b"nonfinancial-control-key").hexdigest(),
        on_http_attempt=kwargs.pop("on_http_attempt", Mock()),
        **kwargs,
    )


@pytest.fixture
def control(tmp_path, monkeypatch):
    loader = Mock(return_value="nonfinancial-control-key")
    monkeypatch.setattr(symbology, "_load_api_key", loader)
    guard = LockboxGuard(tmp_path / "usage.jsonl", ledger=symbology.SYMBOLOGY_LEDGER)
    return guard, loader


@pytest.mark.parametrize("purpose", ["dev_smoke", "discovery", "validation", "forward_accrual"])
def test_wrong_purpose_refuses_before_credentials(control, purpose):
    guard, loader = control
    with pytest.raises(symbology.SymbologyRefused, match="research_scope"):
        symbology.fetch_symbology(
            symbology.SymbologyRequest(),
            guard=guard,
            purpose=purpose,
            arm_id=symbology.SYMBOLOGY_ARM,
            expected_api_key_sha256="",
            on_http_attempt=Mock(),
        )
    loader.assert_not_called()
    assert not guard.usage_log_path.exists()


def test_wrong_ledger_refuses_before_credentials(control):
    guard, loader = control
    guard = LockboxGuard(guard.usage_log_path, ledger=INTRADAY_CAMPAIGN_LEDGER_V1)
    with pytest.raises(symbology.SymbologyRefused, match="research_scope"):
        _fetch(guard)
    loader.assert_not_called()


def test_key_drift_stops_before_transport(control):
    guard, loader = control
    loader.return_value = "changed-control-key"
    witness = Mock()
    with pytest.raises(symbology.SymbologyRefused, match="credential_context"):
        _fetch(guard, on_http_attempt=witness)
    witness.assert_not_called()


def test_exact_free_post_with_guard_before_credentials_and_one_entry(control):
    guard, loader = control
    witness = Mock()
    observed = []

    def load_after_guard():
        entry = json.loads(guard.usage_log_path.read_bytes())
        assert entry["dataset"] == symbology.SYMBOLOGY_DATASET
        assert entry["start"] == entry["end"] == "2018-05-01"
        return "nonfinancial-control-key"

    loader.side_effect = load_after_guard

    def handle(request):
        observed.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(_document()).encode()),
        )

    raw = _fetch(guard, on_http_attempt=witness, transport=httpx.MockTransport(handle))
    assert raw.body == json.dumps(_document()).encode()
    assert len(observed) == 1
    assert observed[0].method == "POST"
    assert str(observed[0].url) == "https://hist.databento.com/v0/symbology.resolve"
    from urllib.parse import parse_qs

    assert parse_qs(observed[0].content.decode()) == {
        "dataset": ["XNAS.ITCH"],
        "symbols": ["ALL_SYMBOLS"],
        "stype_in": ["raw_symbol"],
        "stype_out": ["instrument_id"],
        "start_date": ["2018-05-01"],
        "end_date": ["2018-05-02"],
    }
    witness.assert_called_once_with()


@pytest.mark.parametrize("status", [302, 403, 429, 500])
def test_http_refusal_returned_raw_without_retry_or_fallback(control, status):
    guard, _ = control
    witness = Mock()
    raw = _fetch(
        guard,
        on_http_attempt=witness,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                status,
                headers={"location": "https://other.invalid"},
                stream=httpx.ByteStream(b"nonfinancial refusal"),
            )
        ),
    )
    assert raw.body == b"nonfinancial refusal"
    assert raw.http_status == status
    witness.assert_called_once_with()
    with pytest.raises(symbology.SymbologyRefused, match="http_refused"):
        symbology.parse_symbology(raw, symbology.SymbologyRequest())


def test_oversize_interval_catalog_refuses_without_repair():
    document = _document()
    document["result"]["CONTROL"] *= 100001
    with pytest.raises(symbology.SymbologyRefused, match="response_byte_limit"):
        symbology.parse_symbology(_raw(document), symbology.SymbologyRequest())


def test_symbol_limit_refuses_before_snapshot():
    document = _document()
    document["result"] = {f"CONTROL_{i}": [] for i in range(100001)}
    with pytest.raises(symbology.SymbologyRefused):
        symbology.parse_symbology(_raw(document), symbology.SymbologyRequest())


@pytest.mark.parametrize("change", ["arm", "final"])
def test_unapproved_arm_or_final_review_refuses_before_key(control, change):
    guard, loader = control
    with pytest.raises(symbology.SymbologyRefused, match="research_scope"):
        symbology.fetch_symbology(
            symbology.SymbologyRequest(),
            guard=guard,
            purpose="characterization",
            arm_id="other-arm" if change == "arm" else symbology.SYMBOLOGY_ARM,
            final_portfolio_review=change == "final",
            expected_api_key_sha256="",
            on_http_attempt=Mock(),
        )
    loader.assert_not_called()


@pytest.mark.parametrize("change", ["overlap", "not_found", "both", "echo"])
def test_contradictory_identity_metadata_refuses_without_selecting_mapping(change):
    document = _document()
    if change == "overlap":
        document["result"]["CONTROL"].append({"d0": "2018-05-01", "d1": "2018-05-02", "s": "772"})
    elif change == "not_found":
        document["not_found"] = ["CONTROL"]
    elif change == "both":
        document["result"] = {}
        document["partial"] = ["CONTROL"]
        document["not_found"] = ["CONTROL"]
    else:
        document["symbols"] = ["OTHER_CONTROL"]
    with pytest.raises(symbology.SymbologyRefused):
        symbology.parse_symbology(_raw(document), symbology.SymbologyRequest())


def test_all_symbols_echo_retains_observed_metadata_without_completeness_claim():
    document = _document()
    document["symbols"] = ["ALL_SYMBOLS"]
    snapshot = symbology.parse_symbology(_raw(document), symbology.SymbologyRequest())
    assert snapshot.symbols == ("ALL_SYMBOLS",)
    assert snapshot.mappings[0].raw_symbol == "CONTROL"
    assert snapshot.completeness_claim == "NOT_ESTABLISHED"

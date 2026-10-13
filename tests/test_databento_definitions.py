"""Nonfinancial transport and authority controls; no definition acceptance fixture."""

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock
from urllib.parse import parse_qs

import httpx
import pytest

from liq.data.lockbox import INTRADAY_CAMPAIGN_LEDGER_V1, LockboxGuard
from liq.data.providers import databento_definitions as definitions


def _pin(path: Path, body: bytes) -> definitions.AuthorityPin:
    path.write_bytes(body)
    return definitions.AuthorityPin(path, hashlib.sha256(body).hexdigest())


@pytest.fixture
def control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "capture"
    root.mkdir(mode=0o700)
    pins = tuple(
        _pin(tmp_path / name, b"Nonfinancial control document: " + name.encode())
        for name in ("human", "oracle", "quote", "source")
    )
    authority = definitions.DefinitionAuthority(
        execution_id="control-only",
        capture_root=root,
        human=pins[0],
        oracle=pins[1],
        quote=pins[2],
        source_pins=(pins[3],),
        api_key_sha256=hashlib.sha256(b"control-only-key").hexdigest(),
        estimated_usd=Decimal("0.005364418030"),
        estimate_ceiling_usd=Decimal("1"),
        quote_valid_until=datetime.now(UTC) + timedelta(hours=1),
        uncapped_billing_acknowledged=True,
    )
    monkeypatch.setattr(definitions, "_AUTHORITIES", MappingProxyType({"control-only": authority}))
    loader = Mock(return_value="control-only-key")
    monkeypatch.setattr(definitions, "_load_api_key", loader)
    guard = LockboxGuard(root / "lockbox_usage.jsonl", ledger=definitions.DEFINITION_LEDGER)
    return authority, guard, loader


def _fetch(guard: LockboxGuard, **kwargs):
    return definitions.fetch_definitions(
        definitions.DefinitionRequest(),
        authority_id="control-only",
        guard=guard,
        purpose="characterization",
        arm_id=definitions.DEFINITION_ARM,
        **kwargs,
    )


def test_default_registry_refuses_before_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(definitions, "_AUTHORITIES", MappingProxyType({}))
    loader = Mock()
    monkeypatch.setattr(definitions, "_load_api_key", loader)
    guard = LockboxGuard(tmp_path / "usage", ledger=definitions.DEFINITION_LEDGER)
    with pytest.raises(definitions.DefinitionRefused, match="unknown_authority"):
        _fetch(guard)
    loader.assert_not_called()
    assert not guard.usage_log_path.exists()


@pytest.mark.parametrize("purpose", ["dev_smoke", "discovery", "validation", "forward_accrual"])
def test_wrong_purpose_never_loads_credentials(control, purpose):
    _, guard, loader = control
    with pytest.raises(definitions.DefinitionRefused, match="research_scope"):
        definitions.fetch_definitions(
            definitions.DefinitionRequest(),
            authority_id="control-only",
            guard=guard,
            purpose=purpose,
            arm_id=definitions.DEFINITION_ARM,
        )
    loader.assert_not_called()
    assert not guard.usage_log_path.exists()


@pytest.mark.parametrize("change", ["arm", "final", "ledger", "log"])
def test_wrong_research_guard_never_loads_credentials(control, change):
    _, guard, loader = control
    arm = "other-arm" if change == "arm" else definitions.DEFINITION_ARM
    if change == "ledger":
        guard = LockboxGuard(guard.usage_log_path, ledger=INTRADAY_CAMPAIGN_LEDGER_V1)
    if change == "log":
        guard = LockboxGuard(
            guard.usage_log_path.with_name("other"), ledger=definitions.DEFINITION_LEDGER
        )
    with pytest.raises(definitions.DefinitionRefused):
        definitions.fetch_definitions(
            definitions.DefinitionRequest(),
            authority_id="control-only",
            guard=guard,
            purpose="characterization",
            arm_id=arm,
            final_portfolio_review=change == "final",
        )
    loader.assert_not_called()


@pytest.mark.parametrize("change", ["expired", "budget", "negative", "nan", "uncapped", "pin"])
def test_authority_controls_refuse_before_credentials(control, monkeypatch, change):
    authority, guard, loader = control
    changes = {
        "expired": {"quote_valid_until": datetime.now(UTC) - timedelta(seconds=1)},
        "budget": {"estimated_usd": Decimal("1.01")},
        "negative": {"estimated_usd": Decimal("-1")},
        "nan": {"estimated_usd": Decimal("NaN")},
        "uncapped": {"uncapped_billing_acknowledged": False},
        "pin": {},
    }
    authority = replace(authority, **changes[change])
    if change == "pin":
        authority.quote.path.write_bytes(b"drifted nonfinancial control")
    monkeypatch.setattr(definitions, "_AUTHORITIES", MappingProxyType({"control-only": authority}))
    with pytest.raises(definitions.DefinitionRefused):
        _fetch(guard)
    loader.assert_not_called()
    assert not guard.usage_log_path.exists()


@pytest.mark.parametrize("change", ["start", "end", "limit", "dataset"])
def test_request_is_exact_before_any_authority(control, change):
    _, guard, loader = control
    changes = {
        "start": "2018-05-02T00:00:00Z",
        "end": "2018-05-03T00:00:00Z",
        "limit": 999,
        "dataset": "EQUS.MINI",
    }
    with pytest.raises(definitions.DefinitionRefused, match="request_scope"):
        definitions.fetch_definitions(
            replace(definitions.DefinitionRequest(), **{change: changes[change]}),
            authority_id="control-only",
            guard=guard,
            purpose="characterization",
            arm_id=definitions.DEFINITION_ARM,
        )
    loader.assert_not_called()


def test_exact_post_retains_opaque_bytes_and_one_actual_attempt(control):
    authority, guard, loader = control
    seen = []
    body = b"opaque nonfinancial transport control\n"

    def handle(request):
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, stream=httpx.ByteStream(body)
        )

    result = _fetch(guard, transport=httpx.MockTransport(handle))
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == "https://hist.databento.com/v0/timeseries.get_range"
    assert parse_qs(seen[0].content.decode()) == {
        "dataset": ["XNAS.ITCH"],
        "schema": ["definition"],
        "symbols": ["ALL_SYMBOLS"],
        "stype_in": ["raw_symbol"],
        "stype_out": ["instrument_id"],
        "start": ["2018-05-01T00:00:00Z"],
        "end": ["2018-05-02T00:00:00Z"],
        "limit": ["1000"],
        "encoding": ["json"],
        "compression": ["none"],
        "pretty_px": ["false"],
        "pretty_ts": ["false"],
        "map_symbols": ["false"],
    }
    assert seen[0].headers["accept-encoding"] == "identity"
    assert (authority.capture_root / "response.raw").read_bytes() == body
    assert result.response.body == body
    assert "opaque" not in repr(result)
    usage = json.loads(guard.usage_log_path.read_text())
    assert usage["start"] == usage["end"] == "2018-05-01"
    assert usage["dataset"] == definitions.DEFINITION_DATASET
    attempt = json.loads((authority.capture_root / "attempt.json").read_text())
    assert attempt["ordinal"] == 1
    loader.assert_called_once_with()
    with pytest.raises(definitions.DefinitionRefused, match="consumed"):
        _fetch(guard, transport=httpx.MockTransport(handle))
    assert len(seen) == 1


@pytest.mark.parametrize("status", [302, 403, 429, 500])
def test_http_refusal_retains_body_without_redirect_or_retry(control, status):
    authority, guard, _ = control
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(
            status,
            headers={"location": "https://other.invalid"},
            stream=httpx.ByteStream(b"nonfinancial error control"),
        )

    with pytest.raises(definitions.DefinitionRefused, match="http_refused"):
        _fetch(guard, transport=httpx.MockTransport(handle))
    assert len(seen) == 1
    assert (authority.capture_root / "response.raw").read_bytes() == b"nonfinancial error control"


def test_excess_lines_retained_then_refused(control):
    authority, guard, _ = control
    body = b"opaque control\n" * 1001
    with pytest.raises(definitions.DefinitionRefused, match="record_limit"):
        _fetch(
            guard,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=httpx.ByteStream(body))
            ),
        )
    assert (authority.capture_root / "response.raw").read_bytes() == body


def test_key_context_drift_never_enters_transport(control):
    authority, guard, loader = control
    loader.return_value = "different-control-key"
    transport = Mock(spec=httpx.AsyncBaseTransport)
    with pytest.raises(definitions.DefinitionRefused, match="credential_context"):
        _fetch(guard, transport=transport)
    transport.handle_async_request.assert_not_called()
    assert not (authority.capture_root / "attempt.json").exists()


def test_pin_drift_at_transport_entry_refuses_before_delegate(control, monkeypatch):
    authority, guard, loader = control

    def drift():
        authority.oracle.path.write_bytes(b"changed control authority")
        return "control-only-key"

    loader.side_effect = drift
    transport = Mock(spec=httpx.AsyncBaseTransport)
    with pytest.raises(definitions.DefinitionRefused, match="authority_pin"):
        _fetch(guard, transport=transport)
    transport.handle_async_request.assert_not_called()
    assert not (authority.capture_root / "attempt.json").exists()


def test_definition_request_deadline_closes_transport(control, monkeypatch):
    _, guard, _ = control

    class Slow(httpx.AsyncBaseTransport):
        closed = False

        async def handle_async_request(self, request):
            await asyncio.sleep(60)
            return httpx.Response(200)

        async def aclose(self):
            self.closed = True

    transport = Slow()
    monkeypatch.setattr(definitions, "_REQUEST_SECONDS", 0.02)
    with pytest.raises(definitions.DefinitionRefused, match="request_deadline"):
        _fetch(guard, transport=transport)
    assert transport.closed


@pytest.mark.parametrize("header", ["4194305", "-1", "garbage", "999999999999"])
def test_invalid_or_excess_length_refused_without_raw_capture(control, header):
    authority, guard, _ = control
    with pytest.raises(definitions.DefinitionRefused, match="response_byte_limit"):
        _fetch(
            guard,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, headers={"content-length": header}, stream=httpx.ByteStream(b"control")
                )
            ),
        )
    assert not (authority.capture_root / "response.raw").exists()
    assert (authority.capture_root / "attempt.json").exists()


def test_stream_overflow_refused_with_received_byte_count(control):
    authority, guard, _ = control
    body = b"x" * (4 * 1024 * 1024 + 1)
    with pytest.raises(definitions.DefinitionRefused, match="response_byte_limit") as error:
        _fetch(
            guard,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=httpx.ByteStream(body))
            ),
        )
    assert error.value.received_bytes > 4 * 1024 * 1024
    assert not (authority.capture_root / "response.raw").exists()


def test_compressed_response_refused(control):
    _, guard, _ = control
    with pytest.raises(definitions.DefinitionRefused, match="unexpected_content_encoding"):
        _fetch(
            guard,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(b"control")
                )
            ),
        )


def test_publication_rechecks_authority_after_retaining_raw(control):
    authority, guard, _ = control

    def handle(_):
        authority.human.path.write_bytes(b"drifted authority after actual entry")
        return httpx.Response(200, stream=httpx.ByteStream(b"opaque control"))

    with pytest.raises(definitions.DefinitionRefused, match="authority_pin"):
        _fetch(guard, transport=httpx.MockTransport(handle))
    assert (authority.capture_root / "response.raw").read_bytes() == b"opaque control"
    assert (
        json.loads((authority.capture_root / "response.json").read_text())["schema_interpretation"]
        == "UNVERIFIED"
    )


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "empty", "oversize"])
def test_unsafe_authority_files_refuse_before_credentials(control, kind):
    import os

    authority, guard, loader = control
    path = authority.oracle.path
    if kind == "symlink":
        path.unlink()
        path.symlink_to(authority.human.path)
    elif kind == "hardlink":
        os.link(path, path.with_name("alias"))
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        path.write_bytes(b"" if kind == "empty" else b"x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(definitions.DefinitionRefused, match="authority_pin"):
        _fetch(guard)
    loader.assert_not_called()


@pytest.mark.parametrize("change", ["relative_pin", "missing_pins", "naive_clock", "ceiling"])
def test_incomplete_registered_authority_refuses_before_credentials(control, monkeypatch, change):
    authority, guard, loader = control
    changes = {
        "relative_pin": {"quote": replace(authority.quote, path=Path("relative-control"))},
        "missing_pins": {"source_pins": ()},
        "naive_clock": {"quote_valid_until": datetime.now()},
        "ceiling": {"estimate_ceiling_usd": Decimal("NaN")},
    }
    authority = replace(authority, **changes[change])
    monkeypatch.setattr(definitions, "_AUTHORITIES", MappingProxyType({"control-only": authority}))
    with pytest.raises(definitions.DefinitionRefused):
        _fetch(guard)
    loader.assert_not_called()


@pytest.mark.parametrize("change", ["permissions", "symlink", "missing"])
def test_capture_root_refuses_before_credentials(control, change):
    authority, guard, loader = control
    root = authority.capture_root
    if change == "permissions":
        root.chmod(0o755)
    elif change == "symlink":
        moved = root.with_name("other-root")
        root.rename(moved)
        root.symlink_to(moved)
    else:
        root.rmdir()
    with pytest.raises(definitions.DefinitionRefused, match="capture_root"):
        _fetch(guard)
    loader.assert_not_called()


def test_root_drift_at_transport_entry_refuses_before_delegate(control):
    authority, guard, loader = control

    def drift():
        authority.capture_root.chmod(0o755)
        return "control-only-key"

    loader.side_effect = drift
    transport = Mock(spec=httpx.AsyncBaseTransport)
    with pytest.raises(definitions.DefinitionRefused, match="capture_root"):
        _fetch(guard, transport=transport)
    transport.handle_async_request.assert_not_called()


def test_root_drift_after_response_refuses_publication(control):
    authority, guard, _ = control

    def handle(_):
        authority.capture_root.chmod(0o755)
        return httpx.Response(200, stream=httpx.ByteStream(b"opaque control"))

    with pytest.raises(definitions.DefinitionRefused, match="capture_root"):
        _fetch(guard, transport=httpx.MockTransport(handle))
    assert not (authority.capture_root / "response.raw").exists()


def test_wrong_parent_authority_digest_refuses_before_credentials(control):
    _, guard, loader = control
    with pytest.raises(definitions.DefinitionRefused, match="parent_authority"):
        _fetch(guard, expected_authority_sha256="0" * 64)
    loader.assert_not_called()


def test_capture_returns_original_held_receipt_independent_of_disk(control):
    authority, guard, _ = control
    capture = _fetch(
        guard,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=httpx.ByteStream(b"opaque control"))
        ),
    )
    original = capture.receipt
    (authority.capture_root / "response.json").write_bytes(b"changed control receipt")
    assert capture.receipt == original
    assert json.loads(original)["body_sha256"] == hashlib.sha256(capture.response.body).hexdigest()


def test_pin_read_allows_access_time_update_without_content_change(control, monkeypatch):
    authority, _, _ = control
    original = definitions.os.fstat
    count = 0

    def access_time_only(descriptor):
        nonlocal count
        info = original(descriptor)
        fields = list(info)
        count += 1
        fields[7] += count
        return definitions.os.stat_result(fields)

    monkeypatch.setattr(definitions.os, "fstat", access_time_only)
    assert definitions._read_pin(authority.quote) == authority.quote.path.read_bytes()

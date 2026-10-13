"""Offline filtered membership and audit contracts, not source-admission evidence."""

import json
from pathlib import Path

import httpx
import pytest
import respx

from liq.data.exceptions import UndeclaredReadError, ValidationError
from liq.data.oanda_instrument_metadata import MATRIX_SYMBOLS, InstrumentMetadataRequest
from liq.data.service import DataService
from liq.data.settings import LiqDataSettings


@pytest.fixture
def service(tmp_path: Path) -> DataService:
    return DataService(
        settings=LiqDataSettings.model_construct(
            data_root=tmp_path,
            oanda_api_key="offline-secret",
            oanda_account_id="offline-account",
            oanda_environment="practice",
        )
    )


def request(**changes: str | None | tuple[str, ...] | bool) -> InstrumentMetadataRequest:
    return InstrumentMetadataRequest.model_validate(
        {
            "attempt_id": "a" * 32,
            "source_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "permit_sha256": "d" * 64,
            "purpose": "discovery",
            "arm_id": "idea_05b_london_fix_source_verification",
            "symbols": MATRIX_SYMBOLS,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"purpose": None},
        {"purpose": "validation"},
        {"arm_id": "other"},
        {"symbols": ("EUR_USD",)},
        {"symbols": (*MATRIX_SYMBOLS, "EUR_USD")},
        {"final_portfolio_review": True},
    ],
)
def test_refuses_changed_scope_before_provider_resolution(
    service: DataService,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, str | None | tuple[str, ...] | bool],
) -> None:
    monkeypatch.setattr(service, "_get_provider", lambda _: pytest.fail("provider resolved"))

    with pytest.raises((UndeclaredReadError, ValidationError)):
        service.probe_oanda_instrument_metadata(request(**changes))

    assert not (service.data_root / "instrument_metadata").exists()


@respx.mock
def test_filtered_request_is_pre_audited_without_historical_period(service: DataService) -> None:
    directory = service.data_root / "instrument_metadata" / ("a" * 32)

    def respond(incoming: httpx.Request) -> httpx.Response:
        receipt = json.loads((directory / "receipt.json").read_bytes())
        audit = json.loads((directory / "request.json").read_bytes())
        assert receipt["attempt_id"] == audit["attempt_id"] == "a" * 32
        assert set(incoming.url.params) == {"instruments"}
        assert incoming.url.params["instruments"] == ",".join(MATRIX_SYMBOLS)
        assert set(incoming.extensions["timeout"].values()) == {30.0}
        assert "start" not in audit and "end" not in audit and "dataset" not in audit
        return httpx.Response(200, json={"instruments": [], "lastTransactionID": "private"})

    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/offline-account/instruments"
    ).mock(side_effect=respond)

    observation = service.probe_oanda_instrument_metadata(request())

    assert observation.category == "OBSERVED"
    assert observation.missing_symbols == MATRIX_SYMBOLS
    assert observation.observed_at is not None
    offset = observation.observed_at.utcoffset()
    assert offset is not None and offset.total_seconds() == 0
    assert route.call_count == 1
    assert not (service.data_root / "lockbox_usage_log.jsonl").exists()
    assert "private" not in (directory / "outcome.json").read_text()
    assert "offline-account" not in "".join(path.read_text() for path in directory.iterdir())


@respx.mock
def test_http_failure_retains_attempt_and_refuses_reentry(service: DataService) -> None:
    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/offline-account/instruments"
    ).mock(return_value=httpx.Response(429, text="offline-secret"))
    observation = service.probe_oanda_instrument_metadata(request())

    with pytest.raises(FileExistsError):
        service.probe_oanda_instrument_metadata(request())

    assert route.call_count == 1
    assert observation.category == "HTTP_ERROR" and observation.observed_at is None
    outcome = json.loads(
        (service.data_root / "instrument_metadata" / ("a" * 32) / "outcome.json").read_bytes()
    )
    assert "failed_at" in outcome and "observed_at" not in outcome["observation"]


def test_audit_write_failure_refuses_before_provider_resolution(
    service: DataService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (service.data_root / "instrument_metadata").write_text("not a directory")
    monkeypatch.setattr(service, "_get_provider", lambda _: pytest.fail("provider resolved"))

    with pytest.raises(OSError):
        service.probe_oanda_instrument_metadata(request())


@respx.mock
def test_redirect_is_not_followed_and_account_id_is_redacted(
    service: DataService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger="httpx")
    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/offline-account/instruments"
    ).mock(return_value=httpx.Response(302, headers={"Location": "https://unapproved.invalid"}))

    observation = service.probe_oanda_instrument_metadata(request())

    assert route.call_count == len(respx.calls) == 1
    assert observation.category == "HTTP_ERROR"
    assert "offline-account" not in caplog.text


@respx.mock
def test_transport_failure_is_not_retried(service: DataService) -> None:
    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/offline-account/instruments"
    ).mock(side_effect=httpx.ConnectError("offline-secret"))

    observation = service.probe_oanda_instrument_metadata(request())

    assert route.call_count == 1
    assert observation.category == "TRANSPORT_ERROR" and observation.observed_at is None
    assert "offline-secret" not in observation.model_dump_json()


@respx.mock
def test_new_audit_directory_entries_are_synced_before_transport(
    service: DataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    synced: set[tuple[int, int]] = set()
    original_fsync = os.fsync

    def track_sync(descriptor: int) -> None:
        identity = os.fstat(descriptor)
        original_fsync(descriptor)
        synced.add((identity.st_dev, identity.st_ino))

    monkeypatch.setattr(os, "fsync", track_sync)

    def respond(_: httpx.Request) -> httpx.Response:
        for parent in (service.data_root, service.data_root / "instrument_metadata"):
            identity = parent.stat()
            assert (identity.st_dev, identity.st_ino) in synced
        return httpx.Response(200, json={"instruments": []})

    respx.get(path__regex=r".*/instruments$").mock(side_effect=respond)
    service.probe_oanda_instrument_metadata(request())


@pytest.mark.parametrize("alias", ["attempt", "parent"])
def test_dangling_audit_symlink_cannot_redirect_one_use_claim(
    service: DataService, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    parent = service.data_root / "instrument_metadata"
    target = service.data_root / "redirected"
    if alias == "attempt":
        parent.mkdir()
        (parent / ("a" * 32)).symlink_to(target, target_is_directory=True)
    else:
        parent.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(service, "_get_provider", lambda _: pytest.fail("provider resolved"))

    with pytest.raises(OSError):
        service.probe_oanda_instrument_metadata(request())

    assert not target.exists()


def test_pre_request_audit_corruption_prevents_provider_resolution(
    service: DataService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from liq.data import oanda_instrument_audit

    original_write = oanda_instrument_audit._write_record

    def corrupt_record(
        directory: oanda_instrument_audit._AuditDirectory, name: str, payload: bytes
    ) -> None:
        original_write(directory, name, payload)
        if name == "request.json":
            (directory.path / name).write_bytes(b"{}")

    monkeypatch.setattr(oanda_instrument_audit, "_write_record", corrupt_record)
    monkeypatch.setattr(service, "_get_provider", lambda _: pytest.fail("provider resolved"))
    with pytest.raises(ValidationError, match="audit records disagree"):
        service.probe_oanda_instrument_metadata(request())


@respx.mock
@pytest.mark.parametrize("replacement", ["directory", "symlink", "directory_symlink"])
def test_post_request_audit_replacement_never_publishes_observation(
    service: DataService, replacement: str
) -> None:
    directory = service.data_root / "instrument_metadata" / ("a" * 32)

    def respond(_: httpx.Request) -> httpx.Response:
        if replacement in {"directory", "directory_symlink"}:
            directory.rename(directory.with_name("original"))
            if replacement == "directory":
                directory.mkdir()
            else:
                directory.symlink_to(directory.with_name("original"), target_is_directory=True)
        else:
            record = directory / "request.json"
            backup = service.data_root / "original-request.json"
            record.rename(backup)
            record.symlink_to(backup)
        return httpx.Response(200, json={"instruments": []})

    route = respx.get(path__regex=r".*/instruments$").mock(side_effect=respond)
    with pytest.raises(ValidationError):
        service.probe_oanda_instrument_metadata(request())
    assert route.call_count == 1
    assert not (directory / "outcome.json").exists()


@respx.mock
def test_post_request_audit_corruption_never_publishes_observation(service: DataService) -> None:
    directory = service.data_root / "instrument_metadata" / ("a" * 32)

    def respond(incoming: httpx.Request) -> httpx.Response:
        (directory / "request.json").write_text("changed")
        return httpx.Response(200, json={"instruments": []})

    route = respx.get(
        "https://api-fxpractice.oanda.com/v3/accounts/offline-account/instruments"
    ).mock(side_effect=respond)

    with pytest.raises(ValidationError, match="audit records disagree"):
        service.probe_oanda_instrument_metadata(request())

    assert route.call_count == 1
    assert (directory / "receipt.json").exists() and not (directory / "outcome.json").exists()

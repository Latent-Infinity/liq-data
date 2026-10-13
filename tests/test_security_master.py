from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from liq.data.exceptions import LockboxViolationError, UndeclaredReadError
from liq.data.lockbox import CLASSICAL_RANKER_REFERENCE_LEDGER_V1, resolve_dataset
from liq.data.providers.databento import DatabentoProvider, DatabentoSchemaError
from liq.data.security_master import SecurityMasterRecord
from liq.data.service import DataService


def metadata_record():
    return {
        "ts_effective": datetime(2008, 1, 2, tzinfo=UTC),
        "ts_record": datetime(2008, 1, 3, tzinfo=UTC),
        "ts_created": datetime(2008, 1, 3, tzinfo=UTC),
        "listing_id": "TEST_LISTING",
        "security_id": "TEST_SECURITY",
        "issuer_id": "TEST_ISSUER",
        "exchange": "TEST_EXCHANGE",
        "local_code": None,
    }


def test_security_master_has_separate_dataset():
    assert (
        resolve_dataset("databento", "TEST", data_kind="security_master")
        == "databento_security_master"
    )
    assert resolve_dataset("databento", "TEST") == "databento_extended_hours"


def test_provider_preserves_reference_identity_and_uses_exclusive_end():
    frame = pd.DataFrame([metadata_record()]).set_index("ts_effective")
    endpoint = MagicMock()
    endpoint.get_range.return_value = frame
    provider = DatabentoProvider(
        api_key="test-only",
        reference_client=SimpleNamespace(security_master=endpoint),
        max_retry_attempts=1,
    )
    records = provider.get_security_master(
        ("TEST",), date(2008, 1, 1), date(2008, 12, 31), max_records=10
    )
    assert len(records) == 1
    assert records[0].listing_id == "TEST_LISTING"
    assert records[0].model_dump()["local_code"] is None
    endpoint.get_range.assert_called_once_with(
        start=date(2008, 1, 1),
        end=date(2009, 1, 1),
        index="ts_effective",
        symbols=["TEST"],
        stype_in="raw_symbol",
        countries=["US"],
    )


def test_provider_refuses_record_limit_without_returning_prefix():
    endpoint = MagicMock()
    endpoint.get_range.return_value = pd.DataFrame(
        [metadata_record(), metadata_record()]
    ).set_index("ts_effective")
    provider = DatabentoProvider(
        api_key="test-only", reference_client=SimpleNamespace(security_master=endpoint)
    )
    with pytest.raises(DatabentoSchemaError, match="record limit"):
        provider.get_security_master(("TEST",), date(2008, 1, 1), date(2008, 12, 31), max_records=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ts_effective", "invalid"),
        ("ts_record", datetime(2008, 1, 1)),
        ("listing_id", ""),
        ("security_id", 1),
        ("issuer_id", " "),
    ],
)
def test_malformed_reference_metadata_is_refused(field, value):
    row = metadata_record()
    row[field] = value
    with pytest.raises(ValidationError):
        SecurityMasterRecord.model_validate(row)


def test_record_normalizes_aware_timestamp_to_utc():
    row = metadata_record()
    row["ts_effective"] = "2008-01-02T01:00:00+01:00"
    assert SecurityMasterRecord.model_validate(row).ts_effective == datetime(2008, 1, 2, tzinfo=UTC)


@pytest.fixture
def service(tmp_path):
    return DataService(data_root=tmp_path, lockbox_ledger=CLASSICAL_RANKER_REFERENCE_LEDGER_V1)


def request(service, **overrides):
    params = {
        "provider": "databento",
        "symbols": ("TEST",),
        "start": date(2008, 1, 1),
        "end": date(2008, 12, 31),
        "max_records": 10,
        "purpose": "characterization",
        "arm_id": "classical_cross_sectional_ranker",
    }
    params.update(overrides)
    return service.fetch_security_master(**params)


@pytest.mark.parametrize(
    "overrides",
    [
        {"purpose": "discovery"},
        {"purpose": "dev_smoke"},
        {"arm_id": "other_arm"},
        {"final_portfolio_review": True},
        {"provider": "alpaca"},
        {"start": date(2007, 12, 31)},
        {"end": date(2019, 1, 1)},
    ],
)
def test_invalid_scope_refuses_before_provider_construction(service, overrides):
    with patch.object(service, "_get_provider") as provider, pytest.raises(LockboxViolationError):
        request(service, **overrides)
    provider.assert_not_called()


def test_undeclared_purpose_refuses_before_provider_construction(service):
    with patch.object(service, "_get_provider") as provider, pytest.raises(UndeclaredReadError):
        request(service, purpose=None)
    provider.assert_not_called()


def test_default_campaign_ledger_does_not_open_reference_data(tmp_path):
    service = DataService(data_root=tmp_path)
    with (
        patch.object(service, "_get_provider") as provider,
        pytest.raises(LockboxViolationError, match="does not declare"),
    ):
        request(service)
    provider.assert_not_called()


def test_guarded_reference_request_delegates_once(service):
    provider = SimpleNamespace(get_security_master=MagicMock(return_value=[]))
    with patch.object(service, "_get_provider", return_value=provider):
        assert request(service) == []
    provider.get_security_master.assert_called_once_with(
        ("TEST",), date(2008, 1, 1), date(2008, 12, 31), max_records=10
    )


@pytest.mark.parametrize("symbols", [(), ("",), ("TEST", "TEST"), ("ALL_SYMBOLS",)])
def test_invalid_symbol_scope_refuses_before_provider_construction(service, symbols):
    with patch.object(service, "_get_provider") as provider, pytest.raises(ValueError):
        request(service, symbols=symbols)
    provider.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_records": True},
        {"max_records": 0},
        {"start": date(2009, 1, 1)},
        {"end": date.max},
        {"start": datetime(2008, 1, 1, tzinfo=UTC)},
    ],
)
def test_invalid_request_bounds_refuse_before_provider_construction(service, overrides):
    with patch.object(service, "_get_provider") as provider, pytest.raises(ValueError):
        request(service, **overrides)
    provider.assert_not_called()


def test_numeric_timestamp_is_not_silently_interpreted_as_epoch():
    row = metadata_record()
    row["ts_effective"] = 1
    with pytest.raises(ValidationError):
        SecurityMasterRecord.model_validate(row)


@pytest.mark.parametrize("effective", ["2007-12-31T23:59:59Z", "2009-01-01T00:00:00Z"])
def test_out_of_window_effective_record_is_refused(effective):
    row = metadata_record()
    row["ts_effective"] = effective
    endpoint = SimpleNamespace(
        get_range=MagicMock(return_value=pd.DataFrame([row]).set_index("ts_effective"))
    )
    provider = DatabentoProvider(
        api_key="TEST", reference_client=SimpleNamespace(security_master=endpoint)
    )
    with pytest.raises(DatabentoSchemaError, match="outside requested interval"):
        provider.get_security_master(
            ("TEST",), date(2008, 1, 1), date(2008, 12, 31), max_records=10
        )


def test_non_frame_response_is_typed_schema_refusal():
    endpoint = SimpleNamespace(get_range=MagicMock(return_value=None))
    provider = DatabentoProvider(
        api_key="TEST", reference_client=SimpleNamespace(security_master=endpoint)
    )
    with pytest.raises(DatabentoSchemaError):
        provider.get_security_master(
            ("TEST",), date(2008, 1, 1), date(2008, 12, 31), max_records=10
        )


def test_empty_provider_range_is_preserved_as_empty():
    endpoint = MagicMock()
    endpoint.get_range.return_value = pd.DataFrame()
    provider = DatabentoProvider(
        api_key="test-only", reference_client=SimpleNamespace(security_master=endpoint)
    )
    assert (
        provider.get_security_master(
            ("TEST",), date(2008, 1, 1), date(2008, 12, 31), max_records=10
        )
        == []
    )


def test_provider_authorization_refusal_is_not_retried():
    endpoint = MagicMock()
    endpoint.get_range.side_effect = PermissionError("reference access refused")
    provider = DatabentoProvider(
        api_key="test-only",
        reference_client=SimpleNamespace(security_master=endpoint),
        max_retry_attempts=1,
    )
    with pytest.raises(PermissionError):
        provider.get_security_master(
            ("TEST",), date(2008, 1, 1), date(2008, 12, 31), max_records=10
        )
    assert endpoint.get_range.call_count == 1

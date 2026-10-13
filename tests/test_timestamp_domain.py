import hashlib
import json
import shutil
from contextlib import closing
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from liq.data.exceptions import LockboxViolationError, SchemaValidationError, UndeclaredReadError
from liq.data.service import DataService
from liq.data.timestamp_domain import TimestampDomainAccumulator, TimestampDomainRefused
from liq.store.timestamp_metadata import PinnedParquetFile, inspect_timestamp_metadata
from liq.store.timestamp_projection import TimestampProjectionLimits, iter_pinned_timestamps


@pytest.fixture
def domain_source(tmp_path: Path):
    fixture = Path(__file__).parent / "fixtures/approved/khc-timestamps-20230421-20230424.parquet"
    digest = "7920d9eabd8b2a3ce9392ad82815b8a9eb8c52409c10846ffbcfcecb5fbc49bc"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == digest
    root = tmp_path / "data"
    relative = "tradestation/KHC/bars/1m/year=2023/month=04/approved.parquet"
    target = root / relative
    target.parent.mkdir(parents=True)
    shutil.copyfile(fixture, target)
    identity = PinnedParquetFile(relative, digest)
    expected = inspect_timestamp_metadata(
        root, (identity,), max_files=1, max_file_bytes=100000, max_row_groups=1
    )
    service = DataService(data_root=root, research_log_root=tmp_path / "logs")
    return service, expected


def audit(service, expected, **overrides):
    kwargs = {
        "expected": expected,
        "purpose": "dev_smoke",
        "arm_id": "timestamp-domain-approved-fixture",
        "max_files": 1,
        "max_rows": 449,
        "limits": TimestampProjectionLimits(100000, 449, 1, 128),
    }
    kwargs.update(overrides)
    return service.audit_pinned_timestamps(
        "tradestation", "KHC", "1m", date(2023, 4, 21), date(2023, 4, 24), **kwargs
    )


def test_full_approved_timestamp_domain(domain_source) -> None:
    service, expected = domain_source
    result = audit(service, expected)
    assert result.rows == 449
    assert result.files == 1
    assert result.utc_date_counts == (("2023-04-21", 60), ("2023-04-24", 389))
    assert result.within_group_descents == 0
    assert result.group_boundary_descents == 0
    assert result.file_boundary_descents == 0
    assert result.minimum_ns == expected[0].row_groups[0].minimum_ns
    assert result.maximum_ns == expected[0].row_groups[0].maximum_ns
    records = [
        json.loads(line)
        for line in (service._research_log_root / "timestamp_domain_usage.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [entry["event"] for entry in records] == ["attempt", "completed"]
    assert records[-1]["rows"] == 449


@pytest.mark.parametrize(
    "purpose,error", [(None, UndeclaredReadError), ("discovery", LockboxViolationError)]
)
def test_guard_precedes_projection(domain_source, monkeypatch, purpose, error) -> None:
    service, expected = domain_source
    monkeypatch.setattr(
        "liq.data.service.iter_pinned_timestamps", lambda *_: pytest.fail("projection before guard")
    )
    with pytest.raises(error):
        service.audit_pinned_timestamps(
            "tradestation",
            "KHC",
            "1m",
            date(2026, 1, 1),
            date(2026, 1, 2),
            expected=expected,
            purpose=purpose,
            arm_id="timestamp-domain-approved-fixture",
            max_files=1,
            max_rows=449,
            limits=TimestampProjectionLimits(100000, 449, 1, 128),
        )


def test_key_row_budget_precedes_projection(domain_source, monkeypatch) -> None:
    service, expected = domain_source
    monkeypatch.setattr(
        "liq.data.service.iter_pinned_timestamps",
        lambda *_: pytest.fail("projection before budget"),
    )
    with pytest.raises(TimestampDomainRefused, match="row budget"):
        audit(service, expected, max_rows=448)


def test_known_outside_footer_refuses_before_projection(domain_source, monkeypatch) -> None:
    service, expected = domain_source
    monkeypatch.setattr(
        "liq.data.service.iter_pinned_timestamps",
        lambda *_: pytest.fail("known outside footer reached projection"),
    )
    with pytest.raises(
        TimestampDomainRefused, match="declared footer outside requested dates"
    ) as caught:
        service.audit_pinned_timestamps(
            "tradestation",
            "KHC",
            "1m",
            date(2023, 4, 21),
            date(2023, 4, 21),
            expected=expected,
            purpose="dev_smoke",
            arm_id="timestamp-domain-approved-fixture",
            max_files=1,
            max_rows=449,
            limits=TimestampProjectionLimits(100000, 449, 1, 128),
        )
    assert caught.value.identity == expected[0].identity
    assert caught.value.group == 0
    assert caught.value.partial_rows == 0
    records = [
        json.loads(line)
        for line in (service._research_log_root / "timestamp_domain_usage.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [entry["event"] for entry in records] == ["attempt", "refused"]


def test_unlisted_path_precedes_projection(domain_source, monkeypatch) -> None:
    service, expected = domain_source
    (service._data_root / "tradestation/KHC/bars/1m/unlisted").mkdir()
    monkeypatch.setattr(
        "liq.data.service.iter_pinned_timestamps",
        lambda *_: pytest.fail("projection before inventory"),
    )
    with pytest.raises(SchemaValidationError, match="unexpected path"):
        audit(service, expected)


@pytest.mark.parametrize("max_files,max_rows", [(0, 449), (1, 0), (True, 449)])
def test_invalid_budget(domain_source, max_files, max_rows) -> None:
    service, expected = domain_source
    with pytest.raises(TimestampDomainRefused, match="invalid domain budgets"):
        audit(service, expected, max_files=max_files, max_rows=max_rows)


def test_duplicate_identity_and_empty_request(domain_source) -> None:
    service, expected = domain_source
    for declarations in ((), expected + expected):
        with pytest.raises(TimestampDomainRefused, match="file budget or duplicate path"):
            audit(service, declarations, max_files=2, max_rows=898)


def test_wrong_key_precedes_guard(domain_source, monkeypatch) -> None:
    service, expected = domain_source
    other = replace(expected[0], identity=replace(expected[0].identity, path="other/file.parquet"))
    monkeypatch.setattr(service, "_guard_research_read", lambda *_a, **_k: pytest.fail("wrong key"))
    with pytest.raises(ValueError, match="declared key"):
        audit(service, (other,))


def test_empty_decoder_cannot_complete(domain_source, monkeypatch) -> None:
    service, expected = domain_source

    def empty(*_):
        yield from ()

    monkeypatch.setattr("liq.data.service.iter_pinned_timestamps", empty)
    with pytest.raises(TimestampDomainRefused, match="empty physical domain"):
        audit(service, expected)


@pytest.mark.parametrize("when", ["attempt", "completed"])
def test_log_failure_cannot_return_success(domain_source, monkeypatch, when) -> None:
    service, expected = domain_source
    original_open = Path.open
    attempts = 0

    def fail_domain_log(path, *args, **kwargs):
        nonlocal attempts
        if path.name == "timestamp_domain_usage.jsonl":
            attempts += 1
            if when == "attempt" or attempts >= 2:
                raise OSError("Nonfinancial log-failure sentinel")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_domain_log)
    if when == "attempt":
        monkeypatch.setattr(
            "liq.data.service.iter_pinned_timestamps",
            lambda *_: pytest.fail("projection after log refusal"),
        )
    with pytest.raises(OSError, match="log-failure sentinel"):
        audit(service, expected)


def test_observation_refusal_closes_projection(domain_source, monkeypatch) -> None:
    service, expected = domain_source
    closed = []

    def refuse_observation(self, batch):
        raise TimestampDomainRefused(
            batch.identity, batch.group, 0, None, 0, "nonfinancial anomaly sentinel"
        )

    monkeypatch.setattr(TimestampDomainAccumulator, "observe", refuse_observation)

    def projection(*args):
        try:
            yield from iter_pinned_timestamps(*args)
        finally:
            closed.append(True)

    monkeypatch.setattr("liq.data.service.iter_pinned_timestamps", projection)
    with pytest.raises(TimestampDomainRefused, match="nonfinancial anomaly sentinel"):
        service.audit_pinned_timestamps(
            "tradestation",
            "KHC",
            "1m",
            date(2023, 4, 21),
            date(2023, 4, 24),
            expected=expected,
            purpose="dev_smoke",
            arm_id="timestamp-domain-approved-fixture",
            max_files=1,
            max_rows=449,
            limits=TimestampProjectionLimits(100000, 449, 1, 128),
        )
    assert closed == [True]


@pytest.mark.parametrize(
    "state,reason",
    [
        ("rows", "row budget"),
        ("seen", "duplicate physical timestamp"),
        ("dates", "duplicate UTC date"),
    ],
)
def test_control_state_refusal_uses_unchanged_observation(domain_source, state, reason) -> None:
    service, expected = domain_source
    with closing(
        iter_pinned_timestamps(
            service._data_root, expected[0], TimestampProjectionLimits(100000, 449, 1, 128)
        )
    ) as projection:
        batch = next(projection)
    accumulator = TimestampDomainAccumulator(
        date(2023, 4, 21), date(2023, 4, 24), daily=False, max_rows=449
    )
    value = batch.timestamps_ns[0]
    if state == "rows":
        accumulator.rows = 449
    elif state == "seen":
        accumulator._seen.add(value)
    else:
        accumulator.daily = True
        accumulator._dates[value // (86400 * 10**9)] = 1
    with pytest.raises(TimestampDomainRefused, match=reason) as caught:
        accumulator.observe(batch)
    assert caught.value.timestamp_ns == value
    assert caught.value.row_offset == 0


@pytest.mark.parametrize("boundary", ["within_group", "group_boundary", "file_boundary"])
def test_descent_counters_from_control_state(domain_source, boundary) -> None:
    service, expected = domain_source
    with closing(
        iter_pinned_timestamps(
            service._data_root, expected[0], TimestampProjectionLimits(100000, 449, 1, 128)
        )
    ) as projection:
        batch = next(projection)
    accumulator = TimestampDomainAccumulator(
        date(2023, 4, 21), date(2023, 4, 24), daily=False, max_rows=449
    )
    identity = batch.identity
    group = batch.group
    if boundary == "file_boundary":
        identity = replace(identity, path="nonfinancial-control-state")
    elif boundary == "group_boundary":
        group = -1
    accumulator._previous = (identity, group, expected[0].row_groups[0].maximum_ns)
    accumulator.observe(batch)
    result = accumulator.finish()
    assert result.rows == 128
    for name in ("within_group", "group_boundary", "file_boundary"):
        assert getattr(result, f"{name}_descents") == int(name == boundary)


@pytest.mark.parametrize(
    "max_rows,start,end",
    [(0, date(2023, 4, 21), date(2023, 4, 24)), (1, date(2023, 4, 24), date(2023, 4, 21))],
)
def test_invalid_accumulator_scope(max_rows, start, end) -> None:
    with pytest.raises(ValueError, match="invalid timestamp domain"):
        TimestampDomainAccumulator(start, end, daily=False, max_rows=max_rows)


def test_incomplete_projection_cannot_complete(domain_source, monkeypatch) -> None:
    service, expected = domain_source

    def truncated(*args):
        with closing(iter_pinned_timestamps(*args)) as projection:
            yield next(projection)

    monkeypatch.setattr("liq.data.service.iter_pinned_timestamps", truncated)
    with pytest.raises(TimestampDomainRefused, match="physical domain declaration mismatch"):
        audit(service, expected)


def test_anomaly_refusal_stops_before_next_file(domain_source, monkeypatch) -> None:
    service, expected = domain_source
    identity = replace(
        expected[0].identity,
        path=expected[0].identity.path.replace("approved.parquet", "unread.parquet"),
    )
    (service._data_root / identity.path).write_text("Nonfinancial unopened-file sentinel")
    second = replace(expected[0], identity=identity)
    opened = []

    def refuse_observation(self, batch):
        raise TimestampDomainRefused(
            batch.identity, batch.group, 0, None, 0, "nonfinancial anomaly sentinel"
        )

    monkeypatch.setattr(TimestampDomainAccumulator, "observe", refuse_observation)

    def projection(root, item, limits):
        opened.append(item.identity)
        if len(opened) > 1:
            pytest.fail("opened another file after domain refusal")
        yield from iter_pinned_timestamps(root, item, limits)

    monkeypatch.setattr("liq.data.service.iter_pinned_timestamps", projection)
    with pytest.raises(TimestampDomainRefused, match="nonfinancial anomaly sentinel"):
        service.audit_pinned_timestamps(
            "tradestation",
            "KHC",
            "1m",
            date(2023, 4, 21),
            date(2023, 4, 24),
            expected=(*expected, second),
            purpose="dev_smoke",
            arm_id="timestamp-domain-approved-fixture",
            max_files=2,
            max_rows=898,
            limits=TimestampProjectionLimits(100000, 449, 1, 128),
        )
    assert opened == [expected[0].identity]


def test_accumulator_checks_every_unchanged_observation(domain_source) -> None:
    service, expected = domain_source
    with closing(
        iter_pinned_timestamps(
            service._data_root, expected[0], TimestampProjectionLimits(100000, 449, 1, 128)
        )
    ) as projection:
        batch = next(projection)
    accumulator = TimestampDomainAccumulator(
        date(2023, 4, 21), date(2023, 4, 21), daily=False, max_rows=449
    )
    with pytest.raises(TimestampDomainRefused, match="outside declared dates") as caught:
        accumulator.observe(batch)
    assert caught.value.row_offset == 60
    assert caught.value.partial_rows == 60
    assert caught.value.timestamp_ns == batch.timestamps_ns[60]

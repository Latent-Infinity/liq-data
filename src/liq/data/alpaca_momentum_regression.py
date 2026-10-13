"""One authorized, unchanged retained case; no provider or feature computation."""

import os
import stat
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from liq.data.alpaca_daily_regression import _read
from liq.data.alpaca_momentum_fixture import (
    MomentumDailyFixture,
    parse_momentum_calendar,
    parse_momentum_daily,
)
from liq.data.alpaca_momentum_numerical_files import hold_file, write_once
from liq.data.alpaca_momentum_numerical_models import MomentumNumericalRefused
from liq.data.alpaca_momentum_numerical_scope import (
    NumericalBinding,
    check_numerical_guard,
    reserve_numerical_read,
)
from liq.data.alpaca_momentum_source import AlpacaMomentumDailyRequest
from liq.data.lockbox import LockboxGuard


def read_momentum_case(binding: NumericalBinding, guard: LockboxGuard) -> MomentumDailyFixture:
    """Charge before opening pages and preserve original FD witnesses during decode."""
    binding.check()
    request = binding.contract.request
    check_numerical_guard(request, guard)
    usage_root = Path(binding.contract.usage_root)
    log = usage_root / "numerical-lockbox-usage.jsonl"
    if guard.usage_log_path != log:
        raise MomentumNumericalRefused("numerical_log_path")
    row = reserve_numerical_read(binding)
    try:
        descriptor = os.open(
            log,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
    except OSError as exc:
        raise MomentumNumericalRefused("usage_log_open") from exc
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
        or info.st_size > 65536
    ):
        os.close(descriptor)
        raise MomentumNumericalRefused("usage_log_policy")
    with os.fdopen(descriptor, "ab") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    with hold_file(log, 65536) as original:
        original.check()
    guard.assert_period_allowed(
        request.dataset,
        request.start,
        request.end,
        purpose=request.purpose,
        arm_id=request.arm_id,
        final_portfolio_review=request.final_portfolio_review,
    )
    root = Path(binding.contract.fixture_root)
    witnesses = {item.name: item for item in binding.source.source_witnesses}
    if len(witnesses) != len(binding.source.source_witnesses) or not {
        "calendar.json",
        "khc-daily.json",
        "calendar-mapping.json",
        "guard-final.jsonl",
        "worker-receipt.json",
        "worker-started.json",
        "calendar-attempt.json",
        "daily-attempt.json",
    } == set(witnesses):
        raise MomentumNumericalRefused("source_inventory")
    with ExitStack() as stack:
        originals = []
        for witness in binding.source.source_witnesses:
            original = stack.enter_context(hold_file(root / witness.name, 131072, read_body=False))
            original.require_capture(witness)
            originals.append((original, witness))
        calendar_witness = witnesses["calendar.json"]
        calendar_raw = _read(
            root, calendar_witness.name, calendar_witness.sha256, calendar_witness.bytes + 1
        )
        calendar = parse_momentum_calendar(
            calendar_raw, datetime.fromisoformat(binding.source.receipt.calendar_captured_at)
        )
        write_once(
            usage_root, f"{binding.invocation}-daily-opened.json", row.model_dump_json().encode()
        )
        daily_witness = witnesses["khc-daily.json"]
        daily_raw = _read(root, daily_witness.name, daily_witness.sha256, daily_witness.bytes + 1)
        fixture = parse_momentum_daily(
            daily_raw,
            AlpacaMomentumDailyRequest(),
            calendar,
            datetime.fromisoformat(binding.source.receipt.daily_captured_at),
        )
        if (
            len(fixture.original_labels) != request.decoded_daily_records
            or len(calendar.monthly_endpoints) != request.selected_endpoints
        ):
            raise MomentumNumericalRefused("case_record_counts")
        for original, witness in originals:
            original.require_capture(witness)
        binding.check()
        write_once(usage_root, f"{binding.invocation}-decoded.json", row.model_dump_json().encode())
        return fixture

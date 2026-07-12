"""Research lockbox guard: fold-boundary enforcement for research data reads.

Every signal-bearing research read must declare a ``purpose`` and ``arm_id``
and is checked against a :class:`LockboxLedger` of per-dataset fold windows
(discovery / validation / program lockbox / characterization / forward
accrual). Permitted reads are appended to an append-only JSONL usage log;
violations raise :class:`~liq.data.exceptions.LockboxViolationError` before
any data is touched.

Usage::

    from liq.data.lockbox import LockboxGuard

    guard = LockboxGuard(usage_log_path=log_path)
    guard.assert_period_allowed(
        "oanda_fx", date(2020, 1, 1), date(2023, 12, 31),
        purpose="discovery", arm_id="idea_05a",
    )

``DataService.load`` routes reads through this guard automatically whenever a
``purpose`` is declared. Reads without a declared purpose are not research
reads and are neither checked nor logged; results derived from them can never
be cited as research evidence.

Rules enforced (fail-closed):

- Reads touching a dataset's program-lockbox period raise unless the
  human-only ``final_portfolio_review=True`` flag is passed on a lockbox-only
  read (the flag is recorded in the usage log).
- ``discovery`` / ``validation`` reads must fall entirely inside the
  dataset's corresponding window.
- A dataset admits one validation use per arm; a second use (from a new
  guard session) raises ``ValidationReuseError``.
- Datasets may restrict permitted purposes (Databento extended-hours admits
  only ``characterization`` and ``forward_accrual``).
- Datasets with unassigned folds (Binance perp until depth verification)
  reject every research purpose.
- ``dev_smoke`` is allowed anywhere, tagged in the log, and is never
  research evidence.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

from liq.data.exceptions import LockboxViolationError, ValidationReuseError

RESEARCH_PURPOSES = frozenset({"discovery", "validation", "characterization", "forward_accrual"})
PURPOSES = RESEARCH_PURPOSES | {"dev_smoke"}


@dataclass(frozen=True)
class FoldWindows:
    """Fold boundaries for one dataset (all bounds inclusive, UTC dates)."""

    discovery: tuple[date, date] | None = None
    validation: tuple[date, date] | None = None
    lockbox_start: date | None = None
    characterization: tuple[date, date] | None = None
    forward_accrual_start: date | None = None
    allowed_purposes: frozenset[str] | None = None
    """Restrict research purposes for this dataset; ``None`` means any
    purpose whose window is defined. ``dev_smoke`` is always allowed."""


@dataclass(frozen=True)
class LockboxLedger:
    """Versioned mapping of dataset name to fold windows."""

    version: str
    datasets: Mapping[str, FoldWindows]


INTRADAY_CAMPAIGN_LEDGER_V1 = LockboxLedger(
    version="intraday_campaign_v1",
    datasets=MappingProxyType(
        {
            "spy_qqq_ladder_tradestation": FoldWindows(
                discovery=(date(2015, 1, 1), date(2022, 12, 31)),
                validation=(date(2023, 1, 1), date(2024, 12, 31)),
                lockbox_start=date(2025, 1, 1),
            ),
            "tradestation_cohort_1m": FoldWindows(
                discovery=(date(2019, 1, 1), date(2022, 12, 31)),
                validation=(date(2023, 1, 1), date(2024, 12, 31)),
                lockbox_start=date(2025, 1, 1),
            ),
            "oanda_fx": FoldWindows(
                discovery=(date(2020, 1, 1), date(2023, 12, 31)),
                validation=(date(2024, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            "binance_spot": FoldWindows(
                discovery=(date(2020, 1, 1), date(2023, 12, 31)),
                validation=(date(2024, 1, 1), date(2024, 12, 31)),
                lockbox_start=date(2025, 1, 1),
            ),
            # Folds assigned only after perp depth verification passes.
            "binance_perp": FoldWindows(),
            "databento_extended_hours": FoldWindows(
                characterization=(date(2023, 1, 1), date(2025, 12, 31)),
                forward_accrual_start=date(2026, 1, 1),
                allowed_purposes=frozenset({"characterization", "forward_accrual"}),
            ),
        }
    ),
)

_LADDER_SYMBOLS = frozenset({"SPY", "QQQ"})

USAGE_LOG_FILENAME = "lockbox_usage_log.jsonl"


def resolve_dataset(provider: str, symbol: str) -> str | None:
    """Map a provider/symbol pair to its ledger dataset name.

    Returns ``None`` for providers outside the campaign ledger (their reads
    are not fold-governed).
    """
    provider = provider.lower()
    if provider == "tradestation":
        if symbol.upper() in _LADDER_SYMBOLS:
            return "spy_qqq_ladder_tradestation"
        return "tradestation_cohort_1m"
    if provider == "oanda":
        return "oanda_fx"
    if provider == "binance":
        return "binance_spot"
    if provider == "databento":
        return "databento_extended_hours"
    return None


def _require_window(
    window: tuple[date, date] | None,
    dataset: str,
    purpose: str,
    start: date,
    end: date,
) -> None:
    if window is None:
        raise LockboxViolationError(f"dataset '{dataset}' has no {purpose} window assigned")
    lo, hi = window
    if start < lo or end > hi:
        raise LockboxViolationError(
            f"{purpose} read [{start}, {end}] on '{dataset}' is outside the "
            f"{purpose} window [{lo}, {hi}]"
        )


class LockboxGuard:
    """Enforces the lockbox ledger and appends permitted reads to a log.

    One guard instance represents one run/session: an arm's first
    validation-period read in a session consumes its single validation use;
    further validation reads in the same session are part of that use.
    """

    def __init__(
        self,
        usage_log_path: Path,
        ledger: LockboxLedger = INTRADAY_CAMPAIGN_LEDGER_V1,
    ) -> None:
        self.usage_log_path = Path(usage_log_path)
        self.ledger = ledger
        self._session_id = uuid.uuid4().hex
        self._session_validation_uses: set[tuple[str, str]] = set()

    def assert_period_allowed(
        self,
        dataset: str,
        start: date | None,
        end: date | None,
        *,
        purpose: str,
        arm_id: str,
        final_portfolio_review: bool = False,
    ) -> None:
        """Raise unless the read is permitted; append it to the usage log."""
        if purpose not in PURPOSES:
            raise ValueError(f"unknown purpose '{purpose}'; expected one of {sorted(PURPOSES)}")
        if not arm_id:
            raise ValueError("arm_id must be a non-empty string")
        if start is None or end is None:
            raise ValueError(
                "research reads require explicit start and end dates "
                "(unbounded reads cannot be checked against fold windows)"
            )
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        windows = self.ledger.datasets.get(dataset)
        if windows is None:
            raise LockboxViolationError(
                f"unknown dataset '{dataset}'; known datasets: {sorted(self.ledger.datasets)}"
            )

        if purpose != "dev_smoke":
            if final_portfolio_review:
                self._check_final_portfolio_review_read(
                    dataset, windows, start, end, purpose, arm_id
                )
            else:
                self._check_research_read(dataset, windows, start, end, purpose, arm_id)

        self._append_log_entry(dataset, start, end, purpose, arm_id, final_portfolio_review)

    def _check_final_portfolio_review_read(
        self,
        dataset: str,
        windows: FoldWindows,
        start: date,
        end: date,
        purpose: str,
        arm_id: str,
    ) -> None:
        if windows.allowed_purposes is not None and purpose not in windows.allowed_purposes:
            raise LockboxViolationError(
                f"dataset '{dataset}' only admits purposes "
                f"{sorted(windows.allowed_purposes)}, got '{purpose}'"
            )
        if windows.lockbox_start is None or end < windows.lockbox_start:
            self._check_research_read(dataset, windows, start, end, purpose, arm_id)
            return
        if start < windows.lockbox_start:
            raise LockboxViolationError(
                f"final portfolio review reads for '{dataset}' must be bounded "
                f"to the program lockbox [{windows.lockbox_start}, ...]; split "
                "pre-lockbox fold reads into their ordinary declared-purpose loads"
            )

    def _check_research_read(
        self,
        dataset: str,
        windows: FoldWindows,
        start: date,
        end: date,
        purpose: str,
        arm_id: str,
    ) -> None:
        if windows.allowed_purposes is not None and purpose not in windows.allowed_purposes:
            raise LockboxViolationError(
                f"dataset '{dataset}' only admits purposes "
                f"{sorted(windows.allowed_purposes)}, got '{purpose}'"
            )
        if windows.lockbox_start is not None and end >= windows.lockbox_start:
            raise LockboxViolationError(
                f"read [{start}, {end}] on '{dataset}' touches the program "
                f"lockbox (starts {windows.lockbox_start}); loadable only "
                "under final_portfolio_review=True (human-only)"
            )
        if purpose == "discovery":
            _require_window(windows.discovery, dataset, purpose, start, end)
        elif purpose == "validation":
            _require_window(windows.validation, dataset, purpose, start, end)
            self._check_validation_reuse(dataset, arm_id)
        elif purpose == "characterization":
            if windows.allowed_purposes is None or purpose not in windows.allowed_purposes:
                raise LockboxViolationError(
                    f"dataset '{dataset}' does not admit characterization reads"
                )
            _require_window(windows.characterization, dataset, purpose, start, end)
        elif purpose == "forward_accrual":
            if windows.forward_accrual_start is None:
                raise LockboxViolationError(
                    f"dataset '{dataset}' has no forward-accrual window assigned"
                )
            if start < windows.forward_accrual_start:
                raise LockboxViolationError(
                    f"forward-accrual read [{start}, {end}] on '{dataset}' "
                    f"begins before {windows.forward_accrual_start}"
                )

    def _check_validation_reuse(self, dataset: str, arm_id: str) -> None:
        key = (dataset, arm_id)
        if key in self._session_validation_uses:
            return
        for entry in self._read_log():
            if (
                entry.get("purpose") == "validation"
                and entry.get("dataset") == dataset
                and entry.get("arm_id") == arm_id
                and entry.get("session_id") != self._session_id
            ):
                raise ValidationReuseError(
                    f"arm '{arm_id}' already consumed its validation use of "
                    f"'{dataset}' (session {entry.get('session_id')} at "
                    f"{entry.get('ts')}); a second use requires a new "
                    "pre-registration and human sign-off"
                )
        self._session_validation_uses.add(key)

    def _read_log(self) -> list[dict]:
        if not self.usage_log_path.exists():
            return []
        entries = []
        for line in self.usage_log_path.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    def _append_log_entry(
        self,
        dataset: str,
        start: date,
        end: date,
        purpose: str,
        arm_id: str,
        final_portfolio_review: bool,
    ) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "ledger_version": self.ledger.version,
            "dataset": dataset,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "purpose": purpose,
            "arm_id": arm_id,
            "final_portfolio_review": final_portfolio_review,
        }
        self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

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

``DataService`` routes every read that can return market values through this
guard, and the guard is default-deny: a read that declares no ``purpose`` is
refused with
:class:`~liq.data.exceptions.UndeclaredReadError` before any store or provider
is touched. A safety property that depends on every caller remembering an
optional keyword argument is a convention, not a control.

``dev_smoke`` is the declaration for reads that are not research (fixture
builds, ingestion, diagnostics). It is logged, it is never research evidence,
and the component making the read is named in the free-form ``arm_id``.

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
- Research reads must be bounded; an unbounded read cannot be checked against
  a fold window at all. An unbounded read is admissible only for ``dev_smoke``
  and its log entry records the window as unbounded rather than inventing one.

Providers outside the ledger
----------------------------

Not every provider maps to a ledger dataset, and the absence of a mapping used
to mean the read was served without any record at all. It is now a
classification, and there are exactly three of them:

- **Mapped** -- :func:`resolve_dataset` names a dataset; fold windows apply.
- **Ungoverned** -- the provider is in :data:`UNGOVERNED_PROVIDERS`, an explicit
  allow-list carrying a stated reason per entry. The read is served and appended
  to the usage log with ``dataset: null``, ``governed: false``, the provider
  string, and a stable synthetic identity (``ungoverned:<provider>``) that a
  provenance reconciliation can attribute the period to. A null dataset never
  matches a dataset name, so the entry can never supply citable coverage.
- **Unclassified** -- everything else. Nobody decided, so a research read is
  refused with :class:`~liq.data.exceptions.UnclassifiedProviderError`.
  ``dev_smoke`` is still served and still logged, so ingestion, fixture builds
  and diagnostics keep working and stop being invisible.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

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


# Fold boundaries rolled forward so only the most recent ~6 months stay locked
# (lockbox_start 2026-01-01): discovery extends through 2024, 2025 is the single
# recent out-of-sample validation year. Prior arms characterized on their earlier
# sub-windows remain valid; the change applies to new pre-registrations only.
INTRADAY_CAMPAIGN_LEDGER_V1 = LockboxLedger(
    version="intraday_campaign_v1.3",
    datasets=MappingProxyType(
        {
            "spy_qqq_ladder_tradestation": FoldWindows(
                discovery=(date(2015, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            # Month-end SPY/AGG allocation book, fold-governed independently from
            # the SPY/QQQ ladder (AGG daily history begins 2017 on TradeStation).
            "month_end_spy_agg": FoldWindows(
                discovery=(date(2017, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            # S&P 500 index-reconstitution event book, fold-governed
            # independently from the minute cohort (TradeStation daily-bar
            # depth begins 2017).
            "index_recon_sp500": FoldWindows(
                discovery=(date(2017, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            # Non-signal Path D availability audit authorized 2026-08-17.
            # This does not assign discovery/validation folds or authorize a
            # signal-bearing read; only timestamp-level bar coverage may use it.
            "short_interest_bar_coverage": FoldWindows(
                characterization=(date(2021, 6, 15), date(2025, 12, 31)),
                allowed_purposes=frozenset({"characterization"}),
            ),
            "tradestation_cohort_1m": FoldWindows(
                discovery=(date(2019, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            "oanda_fx": FoldWindows(
                discovery=(date(2020, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            # Venue identity, measured rather than assumed: this dataset holds
            # **Binance.US** bars, not Binance.com. Four partitions of the
            # on-disk ``binance`` store -- two symbols, four years, 11,520
            # minutes -- match a live ``api.binance.us`` pull at 100.00% on
            # close and volume, and ``api.binance.com`` returns HTTP 451 from
            # this environment. The stored parquet carries no venue provenance
            # of its own (schema is bars only, no key-value metadata), so the
            # record lives here.
            #
            # The name is therefore misleading and is deliberately left alone:
            # ``binance_spot`` is a frozen join key for existing usage-ledger
            # entries and is cited by name in pre-registrations, so it is
            # annotated rather than corrected. Evidence and reasoning:
            # ``liq-docs/plans/oracle-binance-us-market-identity-2026-09-24.md``.
            # Pooling Binance.com bars into this dataset is market mixing and
            # is prohibited; ``STORAGE_KEY_VENUES`` enforces that at ingestion.
            "binance_spot": FoldWindows(
                discovery=(date(2020, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            "coinbase_spot": FoldWindows(
                discovery=(date(2020, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            # Folds assigned only after perp depth verification passes.
            "binance_perp": FoldWindows(),
            # CME micro crypto futures (@MBT from 2021, @MET from 2022). Depth
            # verified; fold windows frozen 2026-08-05 by the basis-carry
            # pre-registration. Discovery starts 2022 (the
            # common BTC+ETH inception); the ragged @MBT-2021 tail is excluded
            # from pooled discovery, never backfilled (PD-6).
            "tradestation_crypto_futures": FoldWindows(
                discovery=(date(2022, 1, 1), date(2024, 12, 31)),
                validation=(date(2025, 1, 1), date(2025, 12, 31)),
                lockbox_start=date(2026, 1, 1),
            ),
            "databento_extended_hours": FoldWindows(
                characterization=(date(2023, 1, 1), date(2025, 12, 31)),
                forward_accrual_start=date(2026, 1, 1),
                allowed_purposes=frozenset({"characterization", "forward_accrual"}),
            ),
        }
    ),
)

_FCE_WINDOWS = FoldWindows(
    discovery=(date(2020, 1, 1), date(2024, 12, 31)),
    validation=(date(2025, 1, 1), date(2025, 12, 31)),
    lockbox_start=date(2026, 1, 1),
)

FCE_LOCKBOX_LEDGER_V1 = LockboxLedger(
    version="fce_v1_2020_2026",
    datasets=MappingProxyType(
        {
            "spy_qqq_ladder_tradestation": _FCE_WINDOWS,
            "tradestation_cohort_1m": _FCE_WINDOWS,
            "oanda_fx": _FCE_WINDOWS,
            "coinbase_spot": _FCE_WINDOWS,
            "databento_extended_hours": _FCE_WINDOWS,
            "databento_opra_options": _FCE_WINDOWS,
            "fred_macro": _FCE_WINDOWS,
        }
    ),
)

_LADDER_SYMBOLS = frozenset({"SPY", "QQQ"})
_MONTH_END_SYMBOLS = frozenset({"SPY", "AGG"})
_CRYPTO_FUTURES_CONTINUOUS_SYMBOLS = frozenset({"@MBT", "@MET"})

USAGE_LOG_FILENAME = "lockbox_usage_log.jsonl"


def resolve_dataset(provider: str, symbol: str, *, asset_class: str | None = None) -> str | None:
    """Map a provider/symbol pair to its ledger dataset name.

    ``asset_class`` discriminates reads that share a provider but must be
    fold-governed independently. For Databento, an ``"option"`` read resolves
    to the OPRA options dataset so the guard cannot conflate it with an
    equity-bars read. For TradeStation, a ``"future"`` read resolves to the
    crypto-futures dataset for the same reason. The unambiguous continuous
    symbols ``@MBT`` and ``@MET`` also route there when the hint is omitted;
    bare ``MET`` remains an equity unless explicitly declared as a future.
    An ``"index_recon_book"`` read resolves to the index-reconstitution
    dataset from its daily-bar or corporate-actions providers (TradeStation,
    Databento, Alpaca, or SEC EDGAR) so every source shares its fold governance. Any
    other asset class keeps the equity-bars mapping. A
    ``"short_interest_book"`` TradeStation read resolves to the dedicated
    non-signal bar-coverage dataset.

    ``binance`` and ``binance_us`` are one provider class, one ``use_us``
    boolean and two names for one market surface, and they resolve to the one
    dataset that governs it. That is a record of what the bars are, not a
    market-identity assumption: the ``binance_spot`` dataset already holds
    Binance.US bars (see the annotation on its ledger entry). Refusing to map
    was the claim, and it asserted the two strings name different markets.

    Returns ``None`` for providers outside the campaign ledger (their reads
    are not fold-governed).
    """
    provider = provider.lower()
    if provider == "tradestation":
        if asset_class == "future" or symbol.upper() in _CRYPTO_FUTURES_CONTINUOUS_SYMBOLS:
            return "tradestation_crypto_futures"
        if asset_class == "month_end_book" and symbol.upper() in _MONTH_END_SYMBOLS:
            return "month_end_spy_agg"
        if asset_class == "index_recon_book":
            return "index_recon_sp500"
        if asset_class == "short_interest_book":
            return "short_interest_bar_coverage"
        if symbol.upper() in _LADDER_SYMBOLS:
            return "spy_qqq_ladder_tradestation"
        return "tradestation_cohort_1m"
    if provider == "oanda":
        return "oanda_fx"
    if provider in {"binance", "binance_us"}:
        return "binance_spot"
    if provider == "coinbase":
        return "coinbase_spot"
    if provider == "databento":
        if asset_class == "option":
            return "databento_opra_options"
        if asset_class == "index_recon_book":
            return "index_recon_sp500"
        return "databento_extended_hours"
    if provider in {"alpaca", "sec_edgar"} and asset_class == "index_recon_book":
        return "index_recon_sp500"
    return None


#: Every dataset name :func:`resolve_dataset` can return.
#:
#: A ledger dataset outside this set carries fold windows no read can reach, so
#: the governance somebody wrote down governs nothing. ``tests/test_lockbox.py``
#: derives the same set from the resolver's source and asserts equality, so this
#: constant cannot drift away from the function it describes.
RESOLVABLE_DATASETS = frozenset(
    {
        "spy_qqq_ladder_tradestation",
        "month_end_spy_agg",
        "index_recon_sp500",
        "short_interest_bar_coverage",
        "tradestation_cohort_1m",
        "tradestation_crypto_futures",
        "oanda_fx",
        "binance_spot",
        "coinbase_spot",
        "databento_extended_hours",
        "databento_opra_options",
    }
)


def unreachable_datasets(ledger: LockboxLedger) -> frozenset[str]:
    """Datasets a ledger declares that no :func:`resolve_dataset` call returns.

    The resolver is ledger-blind by construction -- it maps provider/symbol
    pairs without knowing which ledger is in force -- so the two can disagree.
    This names the disagreement in the direction the guard cannot raise on at
    read time: the ledger decided a surface is governed and no read can route
    there. A non-empty result is an owed routing decision, not a bug to paper
    over by adding a mapping branch.
    """
    return frozenset(ledger.datasets) - RESOLVABLE_DATASETS


#: Providers deliberately outside fold governance, with the reason for each.
#:
#: Membership is a governance record, not a convenience. Before this list,
#: "ungoverned" meant only that nobody had written a mapping branch, so it could
#: not be told apart from "nobody has classified this provider". An entry here
#: says somebody decided, and says why. Adding one to make a failing test pass
#: inverts that meaning and is prohibited by
#: ``liq-docs/plans/oracle-unmapped-provider-bypass-2026-09-23.md``.
UNGOVERNED_PROVIDERS: Mapping[str, str] = MappingProxyType(
    {
        "fred": (
            "Macro series. INTRADAY_CAMPAIGN_LEDGER_V1 declares no FRED dataset, so "
            "refusing a read the governing ledger has no opinion about would invent "
            "governance at the guard instead of reading it from the ledger. A "
            "risk-free-rate covariate is also not a tradeable instrument surface with "
            "an out-of-sample to contaminate the way an instrument's bars have. "
            "FCE_LOCKBOX_LEDGER_V1 does declare 'fred_macro', on instrument-bar "
            "windows it inherited structurally; wiring it is an open ledger decision "
            "and is not settled here."
        ),
        "polygon": (
            "Constructible through the provider factories, declared by no dataset in "
            "either ledger, and no live read site was found in the workspace. The "
            "entry exists so that the first live Polygon read is a classification "
            "decision rather than a default."
        ),
    }
)

#: Prefix distinguishing a synthetic ungoverned identity from a ledger dataset.
UNGOVERNED_DATASET_PREFIX = "ungoverned:"


def ungoverned_dataset_identity(provider: str) -> str:
    """Stable synthetic identity for reads of an ungoverned provider.

    A ledger entry for an ungoverned read records ``dataset: null`` so that it
    can never match a governed dataset name, and records this identity
    separately so the read is still attributable. Without it a runner that read
    an ungoverned provider had only two options -- name a dataset that does not
    exist, or omit the period entirely -- and omission is what happened.

    The identity is deliberately not a ledger dataset name and carries no fold
    windows. It answers "what did this arm read that nothing governs", never
    "what may this arm cite".
    """
    return f"{UNGOVERNED_DATASET_PREFIX}{provider.lower()}"


@dataclass(frozen=True)
class StorageKeyVenue:
    """The venue a storage key is declared to hold, and why."""

    provider_name: str
    """The ``MarketDataProvider.name`` whose bars this key admits."""

    reason: str
    """Why the key and the venue differ. A bare mapping would restore the
    silence it replaces."""


#: Storage keys whose declared venue is not simply the key's own name.
#:
#: A storage key is a claim about where bars came from, and until this record
#: existed it was only a claim about what the caller typed. ``DataService``
#: builds the key from the caller's provider argument and never from the
#: constructed provider's ``name``, so the venue a fetch actually contacted
#: reached no storage decision at all. One venue could therefore be written
#: under two keys, or two venues under one, with nothing in the store able to
#: tell -- the parquet schema is bars only and carries no venue provenance.
#:
#: The invariant this exists to enforce: *a bar written to the store is
#: reachable under exactly one provider key, and that key is consistent with
#: the venue the provider object actually contacted.* Entries here are the
#: recorded exceptions to key-equals-venue; everything absent defaults to
#: itself, which is the ordinary case and is asserted as such.
#:
#: Membership is a governance record, not a convenience, and carries the same
#: obligation as :data:`UNGOVERNED_PROVIDERS`: an entry says somebody measured
#: something, and says what.
STORAGE_KEY_VENUES: Mapping[str, StorageKeyVenue] = MappingProxyType(
    {
        "binance": StorageKeyVenue(
            provider_name="binance_us",
            reason=(
                "Measured, not assumed: four partitions of this key's tree -- two "
                "symbols, four years, 11,520 minutes -- match a live api.binance.us "
                "pull at 100.00% on close and volume, while api.binance.com returns "
                "HTTP 451 from this environment. The key therefore holds Binance.US "
                "bars and is the one key for that venue. Two consequences follow. A "
                "fetch under BINANCE_USE_US=true writes truthfully here instead of "
                "mislabelling silently, which is what the February 2026 engineering "
                "note predicted and shipped anyway. And a Binance.com fetch under "
                "this key is refused: pooling two venues in one dataset is market "
                "mixing, which the plan of record forbids for this surface in terms. "
                "See liq-docs/plans/oracle-binance-us-market-identity-2026-09-24.md."
            ),
        ),
    }
)


def declared_storage_venue(storage_provider: str) -> str:
    """The provider ``name`` a storage key is declared to hold.

    Defaults to the key itself, so every provider without a recorded exception
    gets the plain rule: the key names the venue. Only a measured divergence
    earns an entry in :data:`STORAGE_KEY_VENUES`.
    """
    key = storage_provider.lower()
    record = STORAGE_KEY_VENUES.get(key)
    return record.provider_name if record is not None else key


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

    @staticmethod
    def _validate_declaration(
        purpose: str,
        arm_id: str,
        start: date | None,
        end: date | None,
    ) -> None:
        """Check the parts of a declaration that no fold window is needed for.

        Shared by the governed and ungoverned paths so that being outside the
        ledger cannot also mean being outside these. The bounds rule is the one
        that matters most here: ``assert_period_allowed`` refuses an unbounded
        research read because it cannot be checked against a fold window, and an
        ungoverned research read is no more checkable for having no window.
        """
        if purpose not in PURPOSES:
            raise ValueError(f"unknown purpose '{purpose}'; expected one of {sorted(PURPOSES)}")
        if not arm_id:
            raise ValueError("arm_id must be a non-empty string")
        if (start is None or end is None) and purpose != "dev_smoke":
            raise ValueError(
                "research reads require explicit start and end dates "
                "(unbounded reads cannot be checked against fold windows)"
            )
        if start is not None and end is not None and start > end:
            raise ValueError(f"start {start} is after end {end}")

    def classify_read(
        self,
        provider: str,
        symbol: str,
        *,
        asset_class: str | None = None,
    ) -> str | None:
        """Resolve a read to a dataset of *this* ledger, or ``None``.

        :func:`resolve_dataset` cannot see which ledger is in force, so its
        answer and the active ledger can disagree. This reconciles them and
        fails loudly when they do: a dataset name the ledger does not declare is
        a configuration fault and is named as one here, before any store or
        provider is touched, rather than travelling onward as if it were
        governed.

        ``None`` means no dataset of this ledger governs the read. It is not by
        itself permission to serve it; the caller classifies the provider as
        ungoverned or unclassified and acts accordingly.
        """
        dataset = resolve_dataset(provider, symbol, asset_class=asset_class)
        if dataset is not None and dataset not in self.ledger.datasets:
            raise LockboxViolationError(
                f"resolver mapped {provider}/{symbol} to dataset '{dataset}', which "
                f"ledger '{self.ledger.version}' does not declare; known datasets: "
                f"{sorted(self.ledger.datasets)}"
            )
        return dataset

    def record_ungoverned_read(
        self,
        provider: str,
        start: date | None,
        end: date | None,
        *,
        purpose: str,
        arm_id: str,
        final_portfolio_review: bool = False,
    ) -> None:
        """Append a usage-log entry for a read no ledger dataset governs.

        There are no fold windows to check, so nothing is enforced beyond the
        declaration itself; what this buys is the record. Every check that does
        apply to a governed read and does not depend on a fold window applies
        here too -- the purpose must be a known one, ``arm_id`` must be
        non-empty, bounds must not be inverted, and a research purpose must be
        bounded. An unbounded read is the read the guard exists to make
        impossible, and being ungoverned does not make one checkable.

        The entry records ``dataset: null`` and ``governed: false`` so it can
        never be mistaken for, or matched against, a governed dataset, plus the
        provider string and its synthetic identity so the read remains
        attributable.
        """
        self._validate_declaration(purpose, arm_id, start, end)
        self._append_log_entry(
            None, start, end, purpose, arm_id, final_portfolio_review, provider=provider
        )

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
        """Raise unless the read is permitted; append it to the usage log.

        An unbounded read (``start`` or ``end`` omitted) is admissible only for
        ``dev_smoke``, which is fold-exempt anyway, and its log entry records
        the window as unbounded rather than inventing one. Every research
        purpose still requires explicit bounds, because an unbounded read
        cannot be checked against a fold window at all.
        """
        self._validate_declaration(purpose, arm_id, start, end)

        windows = self.ledger.datasets.get(dataset)
        if windows is None:
            raise LockboxViolationError(
                f"unknown dataset '{dataset}'; known datasets: {sorted(self.ledger.datasets)}"
            )

        if purpose != "dev_smoke":
            # Every research purpose was required to supply bounds above, so
            # the fold checks below always receive concrete dates.
            checked_start = cast(date, start)
            checked_end = cast(date, end)
            if final_portfolio_review:
                self._check_final_portfolio_review_read(
                    dataset, windows, checked_start, checked_end, purpose, arm_id
                )
            else:
                self._check_research_read(
                    dataset, windows, checked_start, checked_end, purpose, arm_id
                )

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

    def guarded_windows(
        self,
        *,
        arm_id: str,
        dataset: str,
        current_session_only: bool = False,
    ) -> list[tuple[str, str]]:
        """Citable ``(start, end)`` windows logged for one arm and dataset.

        Only research purposes are returned; ``dev_smoke`` is deliberately
        excluded because it is never evidence. Requiring ``dataset`` prevents a
        guarded read on one market from covering a provenance claim about
        another -- and, since an ungoverned read logs ``dataset: null``, it also
        prevents an ungoverned read from covering any governed claim at all.
        """
        return self._logged_windows(
            arm_id=arm_id,
            current_session_only=current_session_only,
            matches=lambda entry: entry.get("dataset") == dataset,
        )

    def ungoverned_windows(
        self,
        *,
        arm_id: str,
        provider: str,
        current_session_only: bool = False,
    ) -> list[tuple[str, str]]:
        """``(start, end)`` windows logged for one arm's ungoverned reads.

        Deliberately a separate method from :meth:`guarded_windows`, which
        answers "what may this arm cite". This one answers "what did this arm
        read that nothing governs", keyed by the synthetic
        ``ungoverned:<provider>`` identity rather than any dataset name, so a
        provenance reconciliation can attribute the period honestly without the
        window ever standing in for fold clearance.

        ``dev_smoke`` is excluded for the same reason it is excluded above: it
        is never evidence, so it has no place in a provenance claim.
        """
        identity = ungoverned_dataset_identity(provider)
        return self._logged_windows(
            arm_id=arm_id,
            current_session_only=current_session_only,
            matches=lambda entry: (
                entry.get("dataset") is None and entry.get("ungoverned_dataset") == identity
            ),
        )

    def _logged_windows(
        self,
        *,
        arm_id: str,
        current_session_only: bool,
        matches: Callable[[dict], bool],
    ) -> list[tuple[str, str]]:
        """Bounded research windows for one arm, narrowed by ``matches``."""
        windows: list[tuple[str, str]] = []
        for entry in self._read_log():
            if entry.get("arm_id") != arm_id:
                continue
            if not matches(entry):
                continue
            if current_session_only and entry.get("session_id") != self._session_id:
                continue
            if entry.get("purpose") not in RESEARCH_PURPOSES:
                continue
            start = entry.get("start")
            end = entry.get("end")
            if isinstance(start, str) and isinstance(end, str):
                windows.append((start, end))
        return windows

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
        dataset: str | None,
        start: date | None,
        end: date | None,
        purpose: str,
        arm_id: str,
        final_portfolio_review: bool,
        *,
        provider: str | None = None,
    ) -> None:
        """Append one permitted read to the usage log.

        A bounded governed read produces exactly the entry shape every
        historical entry has. An unbounded ``dev_smoke`` read records ``null``
        bounds and adds an explicit ``unbounded`` marker, so a later
        reconciliation reads what the read actually had rather than a window
        invented to satisfy a check.

        A ``None`` dataset marks a read no ledger dataset governs. It adds
        ``governed: false``, the provider string as the caller passed it, and
        the lowercased synthetic identity. ``dataset is None`` and
        ``governed is False`` are the same fact recorded twice, once so that
        every window lookup keyed on a dataset name misses, and once so that a
        reader of the log does not have to infer it from an absence.
        """
        entry: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "ledger_version": self.ledger.version,
            "dataset": dataset,
            "start": start.isoformat() if start is not None else None,
            "end": end.isoformat() if end is not None else None,
            "purpose": purpose,
            "arm_id": arm_id,
            "final_portfolio_review": final_portfolio_review,
        }
        if dataset is None:
            entry["governed"] = False
            entry["provider"] = provider
            entry["ungoverned_dataset"] = ungoverned_dataset_identity(provider or "")
        if start is None or end is None:
            entry["unbounded"] = True
        self.usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

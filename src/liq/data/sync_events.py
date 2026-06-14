"""Event names and lock-related exceptions for ``DataService.sync(...)``.

Centralizing these as module-level constants gives us one canonical
spelling per event so the log catalog, emitters, and structured-log
parsers can never disagree. Each constant's value matches the ``event=``
field documented in ``docs/logging.md``.

The :class:`SyncLockedError` lives here because the lock is owned by
``DataService.sync`` rather than by any single provider — exposing it
from the same module the lock contract belongs to keeps the import
surface tidy.
"""

from __future__ import annotations

from liq.data.exceptions import DataError

# ----- event names -----------------------------------------------------------

EVENT_UNIVERSE_RESOLVED = "universe_resolved"
"""A universe was resolved as-of a date prior to ingestion."""

EVENT_PIT_WARNING = "pit_warning"
"""Resolution produced a non-PIT result (e.g., stub-source composite).

Sync proceeds; downstream sweeps refuse non-PIT universes separately.
"""

EVENT_MANIFEST_GAP_DETECTED = "manifest_gap_detected"
"""Per-symbol gap-detection output. Emitted only when gaps > 0."""

EVENT_MANIFEST_RANGE_APPENDED = "manifest_range_appended"
"""A successful provider fetch produced a new manifest range."""

EVENT_MANIFEST_ROLLBACK = "manifest_rollback"
"""A transaction exited with an exception; the manifest claim was undone."""

EVENT_SYNC_STARTED = "sync_started"
"""``DataService.sync(...)`` was invoked. Fires exactly once per call."""

EVENT_SYMBOL_STARTED = "symbol_started"
"""A new symbol's per-symbol fetch loop began. Fires once per symbol
that has at least one gap to fetch — symbols already fully covered
are silent."""

EVENT_BATCH_SUBMITTED = "batch_submitted"
"""A new Databento batch job was submitted (no prior marker found)."""

EVENT_BATCH_RESUMED = "batch_resumed"
"""A prior batch job's marker was found and is being reused — no
re-submission, no re-billing."""

EVENT_BATCH_POLLING = "batch_polling"
"""A batch job is being polled for terminal state. Emitted once per
poll tick so a long-running job is observable rather than appearing
to hang."""

EVENT_BATCH_DOWNLOAD_STARTED = "batch_download_started"
"""The batch job reached terminal state; the download is about to
start. Useful checkpoint between "job finished at venue" and "files
on disk locally."""

EVENT_SYMBOL_COMPLETED = "symbol_completed"
"""A symbol finished all its gaps successfully. Carries row count
for the operator's tally."""

EVENT_SYMBOL_FAILED = "symbol_failed"
"""A symbol's fetch loop raised before completing all gaps. The
manifest claim was rolled back; the exception propagates after the
event."""

EVENT_SYNC_COMPLETED = "sync_completed"
"""``DataService.sync(...)`` finished normally. Fires exactly once
per call. Mirrors ``sync_started``."""


# ----- exceptions ------------------------------------------------------------


class SyncLockedError(DataError):
    """A concurrent sync holds the lock for this (provider, dataset,
    timeframe, universe) tuple. Raised when ``lock_timeout`` elapses
    before the lock can be acquired."""


__all__ = [
    "EVENT_BATCH_DOWNLOAD_STARTED",
    "EVENT_BATCH_POLLING",
    "EVENT_BATCH_RESUMED",
    "EVENT_BATCH_SUBMITTED",
    "EVENT_MANIFEST_GAP_DETECTED",
    "EVENT_MANIFEST_RANGE_APPENDED",
    "EVENT_MANIFEST_ROLLBACK",
    "EVENT_PIT_WARNING",
    "EVENT_SYMBOL_COMPLETED",
    "EVENT_SYMBOL_FAILED",
    "EVENT_SYMBOL_STARTED",
    "EVENT_SYNC_COMPLETED",
    "EVENT_SYNC_STARTED",
    "EVENT_UNIVERSE_RESOLVED",
    "SyncLockedError",
]

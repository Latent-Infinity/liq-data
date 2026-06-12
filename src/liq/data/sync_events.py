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


# ----- exceptions ------------------------------------------------------------


class SyncLockedError(DataError):
    """A concurrent sync holds the lock for this (provider, dataset,
    timeframe, universe) tuple. Raised when ``lock_timeout`` elapses
    before the lock can be acquired."""


__all__ = [
    "EVENT_MANIFEST_GAP_DETECTED",
    "EVENT_MANIFEST_RANGE_APPENDED",
    "EVENT_MANIFEST_ROLLBACK",
    "EVENT_PIT_WARNING",
    "EVENT_UNIVERSE_RESOLVED",
    "SyncLockedError",
]

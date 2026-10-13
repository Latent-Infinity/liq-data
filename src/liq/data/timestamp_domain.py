import hashlib
from dataclasses import dataclass
from datetime import date, timedelta

from liq.data.exceptions import SchemaValidationError
from liq.store.timestamp_metadata import PinnedParquetFile
from liq.store.timestamp_projection import PinnedTimestampBatch

_DAY_NS = 86400 * 10**9
_EPOCH = date(1970, 1, 1)


@dataclass(frozen=True, slots=True)
class TimestampDomainRefused(SchemaValidationError):
    identity: PinnedParquetFile | None
    group: int | None
    row_offset: int
    timestamp_ns: int | None
    partial_rows: int
    reason: str

    def __str__(self) -> str:
        path = self.identity.path if self.identity else "<request>"
        return (
            f"{path} group={self.group} row={self.row_offset} "
            f"timestamp_ns={self.timestamp_ns} partial_rows={self.partial_rows}: {self.reason}"
        )


@dataclass(frozen=True, slots=True)
class TimestampDomainAudit:
    files: int
    rows: int
    minimum_ns: int
    maximum_ns: int
    utc_date_counts: tuple[tuple[str, int], ...]
    within_group_descents: int
    group_boundary_descents: int
    file_boundary_descents: int
    timestamp_sha256: str


class TimestampDomainAccumulator:
    def __init__(self, start: date, end: date, *, daily: bool, max_rows: int) -> None:
        if start > end or type(max_rows) is not int or max_rows <= 0:
            raise ValueError("invalid timestamp domain bounds or row budget")
        self.lower = (start - _EPOCH).days * _DAY_NS
        self.upper = ((end - _EPOCH).days + 1) * _DAY_NS
        self.daily = daily
        self.max_rows = max_rows
        self.rows = 0
        self._seen: set[int] = set()
        self._dates: dict[int, int] = {}
        self._files: set[PinnedParquetFile] = set()
        self._previous: tuple[PinnedParquetFile, int, int] | None = None
        self._minimum: int | None = None
        self._maximum: int | None = None
        self._within_group_descents = 0
        self._group_boundary_descents = 0
        self._file_boundary_descents = 0
        self._digest = hashlib.sha256()

    def observe(self, batch: PinnedTimestampBatch) -> None:
        for index, value in enumerate(batch.timestamps_ns):
            reason = None
            day = value // _DAY_NS
            if self.rows >= self.max_rows:
                reason = "physical key row budget exceeded"
            elif not self.lower <= value < self.upper:
                reason = "physical timestamp outside declared dates"
            elif value in self._seen:
                reason = "duplicate physical timestamp"
            elif self.daily and day in self._dates:
                reason = "duplicate UTC date in daily key"
            if reason is not None:
                raise TimestampDomainRefused(
                    batch.identity, batch.group, batch.row_offset + index, value, self.rows, reason
                )
            if self._previous is not None and value < self._previous[2]:
                if batch.identity != self._previous[0]:
                    self._file_boundary_descents += 1
                elif batch.group != self._previous[1]:
                    self._group_boundary_descents += 1
                else:
                    self._within_group_descents += 1
            self._previous = (batch.identity, batch.group, value)
            self._seen.add(value)
            self._dates[day] = self._dates.get(day, 0) + 1
            self._files.add(batch.identity)
            self._minimum = value if self._minimum is None else min(self._minimum, value)
            self._maximum = value if self._maximum is None else max(self._maximum, value)
            self._digest.update(f"{value}\n".encode("ascii"))
            self.rows += 1

    def finish(self) -> TimestampDomainAudit:
        if self._minimum is None or self._maximum is None:
            raise TimestampDomainRefused(None, None, 0, None, self.rows, "empty physical domain")
        return TimestampDomainAudit(
            files=len(self._files),
            rows=self.rows,
            minimum_ns=self._minimum,
            maximum_ns=self._maximum,
            utc_date_counts=tuple(
                ((_EPOCH + timedelta(days=day)).isoformat(), count)
                for day, count in sorted(self._dates.items())
            ),
            within_group_descents=self._within_group_descents,
            group_boundary_descents=self._group_boundary_descents,
            file_boundary_descents=self._file_boundary_descents,
            timestamp_sha256=self._digest.hexdigest(),
        )

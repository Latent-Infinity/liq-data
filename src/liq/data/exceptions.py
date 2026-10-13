"""Exceptions for liq.data module."""


class DataError(Exception):
    """Base exception for data-related errors."""

    pass


class UndeclaredReadError(DataError):
    """A read reached a market-data entry point without declaring a purpose.

    This is deliberately **not** a :class:`LockboxViolationError` and must never
    become one. The two report different facts:

    - ``UndeclaredReadError`` -- *the caller did not say what the read is for*.
      Nothing has been checked, because there is nothing to check against.
    - ``LockboxViolationError`` -- *the caller said, and the declared read is
      not permitted*. The fold boundaries were evaluated and refused it.

    Keeping them unrelated by inheritance means ``except LockboxViolationError``
    cannot silently swallow a missing declaration, and a reader of a traceback
    can tell the two apart without matching on message text.
    """

    pass


class LockboxViolationError(DataError):
    """A research read violated the lockbox ledger fold boundaries."""

    pass


class UnclassifiedProviderError(DataError):
    """A research read named a provider nobody has classified.

    The third fact, kept unrelated by inheritance to the other two for the same
    reason they are unrelated to each other:

    - ``UndeclaredReadError`` -- *you did not say* what the read is for.
    - ``LockboxViolationError`` -- *you may not*: the declared read was checked
      against fold boundaries and refused.
    - ``UnclassifiedProviderError`` -- *nobody classified this*. The provider is
      neither mapped to a ledger dataset nor recorded on the ungoverned
      allow-list, so there is no decision for the read to be checked against.

    Serving such a read is what made an unmapped provider invisible: it was
    neither fold-checked nor logged, and a declaration on it asserted a research
    fold use that no control would ever see. The read is refused instead, and
    the refusal names the provider so the answer is a classification decision
    rather than a rename.
    """

    pass


class StorageVenueMismatchError(DataError):
    """An ingest would write a venue's bars under a key declared to hold another.

    The fourth fact, and a fact about a *write* rather than a read, which is why
    it is a sibling of the other three rather than a kind of any of them:

    - ``UndeclaredReadError`` -- *you did not say*.
    - ``LockboxViolationError`` -- *you may not*: fold boundaries refused it.
    - ``UnclassifiedProviderError`` -- *nobody classified this provider*.
    - ``StorageVenueMismatchError`` -- *this key does not hold that venue*.

    The storage key is built from the caller's provider string and the venue is
    chosen by settings, so the two can disagree with nothing in the store able
    to tell: bars carry no venue provenance. A key that can hold two venues is a
    dataset whose fold governance describes a market it does not contain, and
    the mislabel is silent by construction. Refusing the write is what stops the
    key being able to lie; the declared venues are recorded in
    :data:`~liq.data.lockbox.STORAGE_KEY_VENUES`.
    """

    pass


class ValidationReuseError(LockboxViolationError):
    """An arm attempted a second validation-period use of a dataset."""

    pass


#: Every refusal a guarded read can raise, as one tuple an ``except`` clause
#: can name.
#:
#: The three above are deliberately siblings, because a traceback has to tell
#: three different facts apart. The cost of that is that a consumer guarding a
#: read has to name all three, and consumers did -- by hand, at thirty-one call
#: sites, in front of the broad handler a universe scan skips missing symbols
#: with. When ``UnclassifiedProviderError`` was added it was a member of none of
#: those hand-written clauses, fell through every one of them into the broad
#: handler behind it, and a governance stop silently became a skipped symbol at
#: every site: exactly the survivorship change the clauses exist to prevent, as
#: a direct consequence of adding the refusal.
#:
#: This is the single source of truth for "a refusal that must never be
#: swallowed", owned by the repository that owns the concept. A consumer writes
#: ``except GOVERNANCE_REFUSALS: raise`` in front of its broad handler; a fourth
#: refusal type then joins every such clause by being added here, and nowhere
#: else. Membership is a governance act, not a code cleanup, and
#: ``StorageVenueMismatchError`` is the first entry added by that route rather
#: than by a hand edit at every site -- which is the shape working as designed.
#:
#: Derived refusals need no entry: ``ValidationReuseError`` is a
#: ``LockboxViolationError`` and the clause already catches it. Ordinary data
#: errors are deliberately absent -- a store miss, an empty partition or a
#: provider gap is a symbol a scan may legitimately skip, and this tuple must
#: not turn those into stopped runs.
GOVERNANCE_REFUSALS: tuple[type[DataError], ...] = (
    UndeclaredReadError,
    LockboxViolationError,
    UnclassifiedProviderError,
    StorageVenueMismatchError,
)


class ProviderError(DataError):
    """Error from a data provider."""

    pass


class RateLimitError(ProviderError):
    """Rate limit exceeded for a provider."""

    pass


class AuthenticationError(ProviderError):
    """Authentication failed for a provider."""

    pass


class ValidationError(DataError):
    """Data validation failed."""

    pass


class ConfigurationError(DataError):
    """Configuration error."""

    pass


class DataQualityError(DataError):
    """Soft/hard data quality failures."""

    pass


class ProviderUnavailableError(ProviderError):
    """Provider unavailable (5xx/network)."""

    pass


class ProviderNoDataError(ProviderError):
    """Provider confirms no data is available for the request."""

    pass


class SchemaValidationError(ValidationError):
    """Schema or field validation failed."""

    pass


class StoredBarsRequiredError(DataError, ValueError):
    """A batched request needs stored bars and cannot aggregate another timeframe."""


class TimestampAuditLimitError(DataError):
    """A timestamp audit exceeded its declared row budget and is incomplete."""

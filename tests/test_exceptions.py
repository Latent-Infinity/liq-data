"""Tests for liq.data.exceptions module."""

import pytest

from liq.data.exceptions import (
    GOVERNANCE_REFUSALS,
    AuthenticationError,
    ConfigurationError,
    DataError,
    DataQualityError,
    LockboxViolationError,
    ProviderError,
    ProviderUnavailableError,
    RateLimitError,
    SchemaValidationError,
    StorageVenueMismatchError,
    UnclassifiedProviderError,
    UndeclaredReadError,
    ValidationError,
    ValidationReuseError,
)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_provider_error_inherits_from_data_error(self) -> None:
        assert issubclass(ProviderError, DataError)

    def test_rate_limit_error_inherits_from_provider_error(self) -> None:
        assert issubclass(RateLimitError, ProviderError)

    def test_authentication_error_inherits_from_provider_error(self) -> None:
        assert issubclass(AuthenticationError, ProviderError)

    def test_validation_error_inherits_from_data_error(self) -> None:
        assert issubclass(ValidationError, DataError)
        assert issubclass(SchemaValidationError, ValidationError)
        assert issubclass(DataQualityError, DataError)
        assert issubclass(ProviderUnavailableError, ProviderError)

    def test_configuration_error_inherits_from_data_error(self) -> None:
        assert issubclass(ConfigurationError, DataError)

    def test_data_error_inherits_from_exception(self) -> None:
        assert issubclass(DataError, Exception)


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_data_error_with_message(self) -> None:
        error = DataError("test message")
        assert str(error) == "test message"

    def test_provider_error_with_message(self) -> None:
        error = ProviderError("provider failed")
        assert str(error) == "provider failed"

    def test_rate_limit_error_with_message(self) -> None:
        error = RateLimitError("rate limit exceeded")
        assert str(error) == "rate limit exceeded"

    def test_authentication_error_with_message(self) -> None:
        error = AuthenticationError("invalid credentials")
        assert str(error) == "invalid credentials"

    def test_provider_unavailable_message(self) -> None:
        error = ProviderUnavailableError("down")
        assert str(error) == "down"


class TestExceptionCatching:
    """Tests for catching exceptions at different levels."""

    def test_catch_provider_error_as_data_error(self) -> None:
        with pytest.raises(DataError):
            raise ProviderError("test")

    def test_catch_rate_limit_as_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            raise RateLimitError("test")

    def test_catch_authentication_as_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            raise AuthenticationError("test")

    def test_catch_validation_as_data_error(self) -> None:
        with pytest.raises(DataError):
            raise ValidationError("test")

    def test_catch_schema_validation_as_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            raise SchemaValidationError("schema")

    def test_catch_data_quality_as_data_error(self) -> None:
        with pytest.raises(DataError):
            raise DataQualityError("dq")

    def test_catch_provider_unavailable_as_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            raise ProviderUnavailableError("down")


class TestGovernanceRefusals:
    """The tuple of refusals a consumer must never swallow.

    Three refusal types, deliberately siblings so that a traceback tells three
    different facts apart, mean every consumer that guards a read has to name
    all three. Thirty-one call sites did exactly that, by hand, and when the
    third type was added it fell through every one of those clauses into the
    broad handler behind them -- the silent survivorship change the clauses
    exist to prevent. :data:`GOVERNANCE_REFUSALS` is the list of record, so a
    consumer writes ``except GOVERNANCE_REFUSALS: raise`` once and a fourth
    refusal type joins every clause by being added here.

    ``StorageVenueMismatchError`` is that fourth type, and it arrived the way
    the shape was designed for: one edit here, none at any call site.

    Membership is pinned by name below. That is the point rather than a
    duplication of it: adding a refusal is a governance act and should be one
    visible edit in the repository that owns the concept, not thirty-one
    invisible ones elsewhere.
    """

    def test_membership_is_the_recorded_refusals(self) -> None:
        expected = (
            UndeclaredReadError,
            LockboxViolationError,
            UnclassifiedProviderError,
            StorageVenueMismatchError,
        )
        assert expected == GOVERNANCE_REFUSALS

    def test_it_is_a_tuple_of_exception_types(self) -> None:
        """An ``except`` clause takes a tuple of classes and nothing else."""
        assert isinstance(GOVERNANCE_REFUSALS, tuple)
        for refusal in GOVERNANCE_REFUSALS:
            assert isinstance(refusal, type)
            assert issubclass(refusal, DataError)

    def test_no_member_is_a_subclass_of_another(self) -> None:
        """Why the tuple has to exist: the three cannot be named by one base.

        A shared base narrower than :class:`DataError` would make
        ``except <base>`` sufficient and this tuple redundant. There is none,
        by design -- each of the three reports a different fact and a reader of
        a traceback must be able to tell them apart.
        """
        for refusal in GOVERNANCE_REFUSALS:
            others = [other for other in GOVERNANCE_REFUSALS if other is not refusal]
            assert not any(issubclass(refusal, other) for other in others)

    @pytest.mark.parametrize("refusal", GOVERNANCE_REFUSALS, ids=lambda cls: cls.__name__)
    def test_the_clause_catches_every_member(self, refusal: type[DataError]) -> None:
        """``except GOVERNANCE_REFUSALS`` is the clause every call site writes."""
        try:
            raise refusal("refused")
        except GOVERNANCE_REFUSALS as caught:
            assert type(caught) is refusal
        else:  # pragma: no cover - the raise above always raises
            pytest.fail(f"{refusal.__name__} escaped the governance clause")

    def test_a_subclass_of_a_member_is_caught_through_its_base(self) -> None:
        """``ValidationReuseError`` is a ``LockboxViolationError`` and needs no entry.

        Recorded so that a later reader meets the reason rather than the
        omission, and does not add every derived refusal to a tuple that
        already covers them.
        """
        assert issubclass(ValidationReuseError, LockboxViolationError)
        with pytest.raises(GOVERNANCE_REFUSALS):
            raise ValidationReuseError("second validation use")

    @pytest.mark.parametrize(
        "tolerated",
        [DataError, ProviderError, ProviderUnavailableError, ValidationError],
        ids=lambda cls: cls.__name__,
    )
    def test_an_ordinary_data_error_is_not_a_refusal(self, tolerated: type[DataError]) -> None:
        """The clause is a governance stop, not a ban on tolerating failures.

        A universe scan that skips a symbol the store cannot serve is correct
        and stays correct. Were ``DataError`` itself in the tuple, every such
        site would turn a missing symbol into a stopped run.
        """
        with pytest.raises(tolerated):
            try:
                raise tolerated("an ordinary failure")
            except GOVERNANCE_REFUSALS:  # pragma: no cover - must not catch
                pytest.fail(f"{tolerated.__name__} was caught as a governance refusal")

    def test_it_is_re_exported_by_the_package(self) -> None:
        """Consumers import from ``liq.data`` or ``liq.data.exceptions``; both answer."""
        import liq.data

        assert liq.data.GOVERNANCE_REFUSALS is GOVERNANCE_REFUSALS
        assert "GOVERNANCE_REFUSALS" in liq.data.__all__

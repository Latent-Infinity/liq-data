"""Tests for liq.data.exceptions module."""

import pytest

from liq.data.exceptions import (
    AuthenticationError,
    ConfigurationError,
    DataError,
    ProviderError,
    RateLimitError,
    DataQualityError,
    ProviderUnavailableError,
    SchemaValidationError,
    ValidationError,
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

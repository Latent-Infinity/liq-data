"""Exceptions for liq.data module."""


class DataError(Exception):
    """Base exception for data-related errors."""

    pass


class LockboxViolationError(DataError):
    """A research read violated the lockbox ledger fold boundaries."""

    pass


class ValidationReuseError(LockboxViolationError):
    """An arm attempted a second validation-period use of a dataset."""

    pass


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

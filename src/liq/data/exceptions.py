"""Exceptions for liq.data module."""


class DataError(Exception):
    """Base exception for data-related errors."""

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


class SchemaValidationError(ValidationError):
    """Schema or field validation failed."""

    pass

"""Retry logic utility with exponential backoff.

This module provides retry decorators for handling transient failures
such as network errors, rate limits, and temporary service unavailability.
Both sync and async versions are provided.

Design Principles:
- SRP: Focused solely on retry logic
- DRY: Single implementation for all retry needs
- KISS: Simple decorator interface

Example:
    from liq.data.retry import retry, async_retry
    from liq.data.exceptions import RateLimitError

    @retry(
        max_retries=3,
        initial_delay=1.0,
        backoff_multiplier=2.0,
        retryable_exceptions=(RateLimitError, ConnectionError)
    )
    def fetch_data():
        # This will automatically retry on RateLimitError or ConnectionError
        # with exponential backoff (1s, 2s, 4s)
        return api.get_data()

    @async_retry(max_retries=3)
    async def fetch_data_async():
        return await api.get_data_async()
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

# Type variables for decorated function return types
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
P = ParamSpec("P")

logger = logging.getLogger(__name__)


def retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff.

    The decorator will catch specified exceptions and retry the function
    up to max_retries times, with exponentially increasing delays between
    attempts.

    Args:
        max_retries: Maximum number of total attempts including the initial call (default: 3).
                    Set to 0 to attempt only once with no retries. Set to 1 for one retry.
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
                           delay = initial_delay * (backoff_multiplier ** attempt)
        retryable_exceptions: Tuple of exception types to retry on.
                             (default: (Exception,) - retries all exceptions)

    Returns:
        Decorated function that implements retry logic

    Example:
        @retry(max_retries=3, initial_delay=0.5, backoff_multiplier=2.0)
        def fetch_data():
            return api.get_data()

        # Will retry up to 3 times with delays: 0.5s, 1.0s, 2.0s

    Raises:
        The last exception raised if all retry attempts are exhausted
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            # Ensure at least one attempt
            for attempt in range(max(1, max_retries)):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(
                            "retry succeeded function=%s attempt=%d",
                            func.__name__,
                            attempt + 1,
                        )
                    return result

                except retryable_exceptions as e:
                    last_exception = e

                    # If this was the last attempt, raise the exception
                    if attempt == max(1, max_retries) - 1:
                        logger.error(
                            "retry exhausted function=%s max_retries=%d error=%s",
                            func.__name__,
                            max_retries,
                            str(e),
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = initial_delay * (backoff_multiplier**attempt)

                    logger.warning(
                        "retry attempt function=%s attempt=%d max_retries=%d delay=%.2f error=%s",
                        func.__name__,
                        attempt + 1,
                        max_retries,
                        delay,
                        str(e),
                    )

                    time.sleep(delay)

            # This should never be reached, but satisfies type checker
            if last_exception:
                raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


def async_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]
]:
    """Async decorator that retries a coroutine with exponential backoff.

    The decorator will catch specified exceptions and retry the coroutine
    up to max_retries times, with exponentially increasing delays between
    attempts.

    Args:
        max_retries: Maximum number of total attempts including the initial call (default: 3).
                    Set to 0 to attempt only once with no retries. Set to 1 for one retry.
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
                           delay = initial_delay * (backoff_multiplier ** attempt)
        retryable_exceptions: Tuple of exception types to retry on.
                             (default: (Exception,) - retries all exceptions)

    Returns:
        Decorated coroutine that implements retry logic

    Example:
        @async_retry(max_retries=3, initial_delay=0.5, backoff_multiplier=2.0)
        async def fetch_data():
            return await api.get_data()

        # Will retry up to 3 times with delays: 0.5s, 1.0s, 2.0s

    Raises:
        The last exception raised if all retry attempts are exhausted
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]],
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None

            # Ensure at least one attempt
            for attempt in range(max(1, max_retries)):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(
                            "async_retry succeeded function=%s attempt=%d",
                            func.__name__,
                            attempt + 1,
                        )
                    return result

                except retryable_exceptions as e:
                    last_exception = e

                    # If this was the last attempt, raise the exception
                    if attempt == max(1, max_retries) - 1:
                        logger.error(
                            "async_retry exhausted function=%s max_retries=%d error=%s",
                            func.__name__,
                            max_retries,
                            str(e),
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = initial_delay * (backoff_multiplier**attempt)

                    logger.warning(
                        "async_retry attempt function=%s attempt=%d max_retries=%d delay=%.2f error=%s",
                        func.__name__,
                        attempt + 1,
                        max_retries,
                        delay,
                        str(e),
                    )

                    await asyncio.sleep(delay)

            # This should never be reached, but satisfies type checker
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in async_retry")  # pragma: no cover

        return wrapper

    return decorator

"""Tests for liq.data.retry module."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from liq.data.retry import async_retry, retry


class TestRetryDecorator:
    """Tests for the retry decorator."""

    def test_success_on_first_attempt(self) -> None:
        """Test function succeeds on first attempt."""
        mock_func = MagicMock(return_value="success")

        @retry(max_retries=3)
        def test_func() -> str:
            return mock_func()

        result = test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retry(self) -> None:
        """Test function succeeds after retrying."""
        call_count = {"n": 0}

        @retry(max_retries=3, initial_delay=0.01)
        def test_func() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("Temporary error")
            return "success"

        result = test_func()

        assert result == "success"
        assert call_count["n"] == 2

    def test_all_retries_exhausted(self) -> None:
        """Test exception raised when all retries exhausted."""

        @retry(max_retries=3, initial_delay=0.01)
        def test_func() -> str:
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            test_func()

    def test_specific_exception_types(self) -> None:
        """Test only specified exceptions trigger retry."""
        call_count = {"n": 0}

        @retry(
            max_retries=3,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        def test_func() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("Retryable")
            return "success"

        result = test_func()

        assert result == "success"
        assert call_count["n"] == 2

    def test_non_retryable_exception_not_caught(self) -> None:
        """Test non-retryable exceptions are raised immediately."""

        @retry(
            max_retries=3,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        def test_func() -> str:
            raise TypeError("Not retryable")

        with pytest.raises(TypeError, match="Not retryable"):
            test_func()

    def test_exponential_backoff(self) -> None:
        """Test exponential backoff between retries."""
        call_times: list[float] = []

        @retry(max_retries=3, initial_delay=0.05, backoff_multiplier=2.0)
        def test_func() -> str:
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Retry")
            return "success"

        result = test_func()

        assert result == "success"
        assert len(call_times) == 3

        # Check delays are approximately correct
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # First delay should be ~0.05s, second ~0.1s
        assert delay1 >= 0.04  # Allow some tolerance
        assert delay2 >= 0.08

    def test_max_retries_zero(self) -> None:
        """Test max_retries=0 still attempts once."""
        call_count = {"n": 0}

        @retry(max_retries=0)
        def test_func() -> str:
            call_count["n"] += 1
            return "success"

        result = test_func()

        assert result == "success"
        assert call_count["n"] == 1

    def test_preserves_function_metadata(self) -> None:
        """Test decorator preserves function name and docstring."""

        @retry(max_retries=3)
        def my_documented_function() -> str:
            """This is my docstring."""
            return "value"

        assert my_documented_function.__name__ == "my_documented_function"
        assert my_documented_function.__doc__ == "This is my docstring."

    def test_passes_arguments(self) -> None:
        """Test decorated function receives arguments correctly."""

        @retry(max_retries=3)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)

        assert result == 5

    def test_passes_kwargs(self) -> None:
        """Test decorated function receives keyword arguments correctly."""

        @retry(max_retries=3)
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")

        assert result == "Hi, World!"


class TestAsyncRetryDecorator:
    """Tests for the async_retry decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        """Test async function succeeds on first attempt."""
        mock_func = AsyncMock(return_value="success")

        @async_retry(max_retries=3)
        async def test_func() -> str:
            return await mock_func()

        result = await test_func()

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retry(self) -> None:
        """Test async function succeeds after retrying."""
        call_count = {"n": 0}

        @async_retry(max_retries=3, initial_delay=0.01)
        async def test_func() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("Temporary error")
            return "success"

        result = await test_func()

        assert result == "success"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self) -> None:
        """Test exception raised when all retries exhausted."""

        @async_retry(max_retries=3, initial_delay=0.01)
        async def test_func() -> str:
            raise ValueError("Persistent error")

        with pytest.raises(ValueError, match="Persistent error"):
            await test_func()

    @pytest.mark.asyncio
    async def test_specific_exception_types(self) -> None:
        """Test only specified exceptions trigger retry."""
        call_count = {"n": 0}

        @async_retry(
            max_retries=3,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        async def test_func() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("Retryable")
            return "success"

        result = await test_func()

        assert result == "success"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_caught(self) -> None:
        """Test non-retryable exceptions are raised immediately."""

        @async_retry(
            max_retries=3,
            initial_delay=0.01,
            retryable_exceptions=(ValueError,),
        )
        async def test_func() -> str:
            raise TypeError("Not retryable")

        with pytest.raises(TypeError, match="Not retryable"):
            await test_func()

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        """Test exponential backoff between retries."""
        call_times: list[float] = []

        @async_retry(max_retries=3, initial_delay=0.05, backoff_multiplier=2.0)
        async def test_func() -> str:
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Retry")
            return "success"

        result = await test_func()

        assert result == "success"
        assert len(call_times) == 3

        # Check delays are approximately correct
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # First delay should be ~0.05s, second ~0.1s
        assert delay1 >= 0.04  # Allow some tolerance
        assert delay2 >= 0.08

    @pytest.mark.asyncio
    async def test_max_retries_zero(self) -> None:
        """Test max_retries=0 still attempts once."""
        call_count = {"n": 0}

        @async_retry(max_retries=0)
        async def test_func() -> str:
            call_count["n"] += 1
            return "success"

        result = await test_func()

        assert result == "success"
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self) -> None:
        """Test decorator preserves function name and docstring."""

        @async_retry(max_retries=3)
        async def my_documented_async_function() -> str:
            """This is my async docstring."""
            return "value"

        assert my_documented_async_function.__name__ == "my_documented_async_function"
        assert my_documented_async_function.__doc__ == "This is my async docstring."

    @pytest.mark.asyncio
    async def test_passes_arguments(self) -> None:
        """Test decorated async function receives arguments correctly."""

        @async_retry(max_retries=3)
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)

        assert result == 5

    @pytest.mark.asyncio
    async def test_passes_kwargs(self) -> None:
        """Test decorated async function receives keyword arguments correctly."""

        @async_retry(max_retries=3)
        async def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await greet("World", greeting="Hi")

        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_uses_asyncio_sleep(self) -> None:
        """Test that async_retry uses asyncio.sleep, not time.sleep."""
        call_count = {"n": 0}

        @async_retry(max_retries=2, initial_delay=0.01)
        async def test_func() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("Retry")
            return "success"

        # This should not block the event loop
        result = await test_func()

        assert result == "success"
        assert call_count["n"] == 2

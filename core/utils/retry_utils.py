import asyncio
import logging
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _calculate_jitter(attempt: int, backoff_in_seconds: float, max_backoff: float) -> float:
    """
    Calculates the sleep time with exponential backoff and full jitter.

    Args:
        attempt (int): The current retry attempt (0-indexed).
        backoff_in_seconds (float): The base backoff time in seconds.
        max_backoff (float): The maximum allowed backoff time in seconds.

    Returns:
        float: The randomized sleep time.
    """
    temp = min(max_backoff, backoff_in_seconds * (2 ** attempt))
    return random.uniform(0, temp)


def retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: float = 1.0,
    max_backoff: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Retry a function with exponential backoff and full jitter.

    This decorator supports both synchronous and asynchronous functions. It uses
    exponential backoff with full jitter to avoid thundering herd problems, and
    allows specifying a maximum backoff limit and specific exceptions to catch.

    Architecture & Usage:
        Use this decorator to wrap network calls, database queries, or flaky API endpoints.
        It enforces deterministic resilience with randomized delays.

    Args:
        retries (int): Maximum number of retry attempts before raising the exception.
        backoff_in_seconds (float): Base backoff time in seconds.
        max_backoff (float): Maximum backoff time in seconds to prevent unbounded sleeps.
        exceptions (tuple[type[Exception], ...]): Tuple of exceptions that should trigger a retry.

    Returns:
        The decorated function.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                attempt = 0
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        if attempt >= retries:
                            logger.error(
                                f"Async function {func.__name__} failed after {retries} retries: {e}"
                            )
                            raise e from None

                        sleep_time = _calculate_jitter(attempt, backoff_in_seconds, max_backoff)
                        logger.warning(
                            f"Async function {func.__name__} failed: {e}. "
                            f"Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{retries})"
                        )
                        await asyncio.sleep(sleep_time)
                        attempt += 1

            return cast(Callable[P, R], async_wrapper)
        else:
            @wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                attempt = 0
                while True:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt >= retries:
                            logger.error(
                                f"Function {func.__name__} failed after {retries} retries: {e}"
                            )
                            raise e from None

                        sleep_time = _calculate_jitter(attempt, backoff_in_seconds, max_backoff)
                        logger.warning(
                            f"Function {func.__name__} failed: {e}. "
                            f"Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{retries})"
                        )
                        time.sleep(sleep_time)
                        attempt += 1

            return sync_wrapper

    return decorator

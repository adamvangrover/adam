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

    Args:
        retries: Maximum number of retry attempts before raising the exception.
        backoff_in_seconds: Base backoff time in seconds.
        max_backoff: Maximum backoff time in seconds to prevent unbounded sleeps.
        exceptions: Tuple of exceptions that should trigger a retry.

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
                            raise

                        # Exponential backoff with full jitter
                        temp = min(max_backoff, backoff_in_seconds * (2 ** attempt))
                        sleep_time = random.uniform(0, temp)

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
                            raise

                        # Exponential backoff with full jitter
                        temp = min(max_backoff, backoff_in_seconds * (2 ** attempt))
                        sleep_time = random.uniform(0, temp)

                        logger.warning(
                            f"Function {func.__name__} failed: {e}. "
                            f"Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{retries})"
                        )
                        time.sleep(sleep_time)
                        attempt += 1

            return sync_wrapper

    return decorator

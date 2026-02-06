import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

class RedundancyOptimizationMixin:
    """
    Mixin to provide graceful fallbacks, caching, and redundancy checks for async agent tasks.
    Implements a 'Stale-While-Revalidate' strategy where possible.
    """

    # Class-level cache to share across instances if needed, or instance-level.
    # Using instance-level for safety, but could be promoted to a shared manager.
    _task_cache: Dict[str, Tuple[float, Any]] = {}  # Key -> (timestamp, result)
    _CACHE_TTL = 300  # 5 minutes default

    async def execute_redundant_task(
        self,
        task_name: str,
        task_func: Callable,
        *args,
        ttl: int = 300,
        use_stale_on_error: bool = True,
        **kwargs
    ) -> Any:
        """
        Executes an async task with caching and redundancy checks.

        Args:
            task_name (str): Unique identifier for the task type.
            task_func (Callable): The async function to execute.
            ttl (int): Time-to-live for cache in seconds.
            use_stale_on_error (bool): If True, returns cached value (even if expired) on execution failure.
        """
        cache_key = self._generate_cache_key(task_name, args, kwargs)
        current_time = time.time()

        # 1. Check Cache
        cached_entry = self._task_cache.get(cache_key)
        if cached_entry:
            timestamp, value = cached_entry
            if current_time - timestamp < ttl:
                logging.info(f"[RedundancyMixin] Cache Hit for {task_name}")
                return value
            else:
                logging.info(f"[RedundancyMixin] Cache Stale for {task_name}. Revalidating...")

        # 2. Execute Task
        try:
            # Check if task_func is a coroutine
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func(*args, **kwargs)
            else:
                result = task_func(*args, **kwargs)

            # Update Cache
            self._task_cache[cache_key] = (current_time, result)
            return result

        except Exception as e:
            logging.error(f"[RedundancyMixin] Task {task_name} failed: {e}")

            # 3. Fallback to Stale
            if use_stale_on_error and cached_entry:
                logging.warning(f"[RedundancyMixin] Returning STALE data for {task_name} due to error.")
                return cached_entry[1]

            # Re-raise if no fallback
            raise e

    def _generate_cache_key(self, name: str, args: tuple, kwargs: dict) -> str:
        """
        Generates a deterministic hash for the task inputs.
        """
        try:
            # specific serialization for JSON stability
            payload = {
                "name": name,
                "args": args,
                "kwargs": kwargs
            }
            # Use default=str to handle non-serializable objects gracefully
            s = json.dumps(payload, sort_keys=True, default=str)
            return hashlib.sha256(s.encode()).hexdigest()
        except Exception as e:
            logging.warning(f"Cache key generation failed: {e}. Using simple name.")
            return name

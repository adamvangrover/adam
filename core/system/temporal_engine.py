# core/system/temporal_engine.py

import asyncio
import logging
import time
from typing import Callable, List, Dict, Any, Awaitable
from datetime import datetime, timedelta

# Configuring specific logger for the Temporal Engine to ensure observability
logger = logging.getLogger("adam.core.system.temporal_engine")

class PulseTask:
    """
    Represents a single recurring task within the Temporal Engine.

    This class encapsulates the logic, schedule, and state of a recurring job.
    It adheres to the "Async First" philosophy by expecting coroutines.
    """
    def __init__(
        self,
        name: str,
        coro_func: Callable[..., Awaitable[Any]],
        interval_seconds: int = 60,
        run_immediately: bool = False,
        **kwargs
    ):
        """
        Args:
            name (str): Unique identifier for the task.
            coro_func (Callable): The async function to execute.
            interval_seconds (int): Frequency of execution in seconds.
            run_immediately (bool): If True, runs once on startup before waiting.
            **kwargs: Arguments to pass to the coroutine.
        """
        self.name = name
        self.coro_func = coro_func
        self.interval = interval_seconds
        self.run_immediately = run_immediately
        self.kwargs = kwargs
        self.last_run: datetime = datetime.min
        self.is_running: bool = False
        self.error_count: int = 0

class TemporalEngine:
    """
    The Async Scheduler for the Adam System.

    Replaces the blocking logic of task_scheduler.py with an asyncio-native approach.
    Allows for 'Heartbeat' tasks that run alongside the main agent loop.
    """
    def __init__(self):
        self.tasks: Dict[str, PulseTask] = {}
        self._stop_event = asyncio.Event()
        logger.info("TemporalEngine initialized. Ready to orchestrate time.")

    def register_task(self, name: str, coro_func: Callable[..., Awaitable[Any]], interval_seconds: int, **kwargs):
        """
        Registers a new task to the schedule.

        Args:
            name: Human-readable name for logs.
            coro_func: The async function to call.
            interval_seconds: How often to call it.
        """
        if name in self.tasks:
            logger.warning(f"Overwriting existing task schedule for: {name}")

        task = PulseTask(name, coro_func, interval_seconds, **kwargs)
        self.tasks[name] = task
        logger.info(f"Task registered: {name} (Interval: {interval_seconds}s)")

    async def _run_task_safe(self, task: PulseTask):
        """
        Executes a task with defensive error handling to prevent loop crashes.
        """
        if task.is_running:
            logger.warning(f"Skipping run for {task.name}: Previous instance still running (Duration > Interval).")
            return

        task.is_running = True
        start_time = time.perf_counter()

        try:
            logger.info(f"PULSE: Executing {task.name}...")
            await task.coro_func(**task.kwargs)
            duration = time.perf_counter() - start_time
            task.last_run = datetime.now()
            task.error_count = 0 # Reset consecutive errors on success
            logger.info(f"PULSE: Finished {task.name} in {duration:.2f}s")

        except Exception as e:
            task.error_count += 1
            logger.error(f"CRITICAL: Task {task.name} failed! Error: {e}", exc_info=True)
            # Add logic here to alert the RedTeamAgent if error_count > threshold

        finally:
            task.is_running = False

    async def start(self):
        """
        Begins the scheduling loop. This is a non-blocking coroutine that should
        be gathered with other main system processes.
        """
        logger.info("TemporalEngine started. The heartbeat is active.")

        # Handle "run_immediately" tasks
        for task in self.tasks.values():
            if task.run_immediately:
                asyncio.create_task(self._run_task_safe(task))

        while not self._stop_event.is_set():
            now = datetime.now()

            for task in self.tasks.values():
                # Calculate simple interval delta
                time_since_last = (now - task.last_run).total_seconds()

                if time_since_last >= task.interval:
                    # Fire and forget (create task) so we don't block checking other schedules
                    # concurrency limitation is handled inside _run_task_safe
                    asyncio.create_task(self._run_task_safe(task))

            # The "Tick". Checks every 1 second.
            # This allows for decent precision without burning CPU.
            await asyncio.sleep(1)

    def stop(self):
        """Signals the engine to shut down gracefully."""
        self._stop_event.set()
        logger.info("TemporalEngine stopping...")

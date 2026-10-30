"""
Background task triggers and event scheduling primitives for Antigravity SDK.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TriggerContext:
    """Context object passed to background trigger task callbacks."""
    agent: Any

    async def send(self, message: str) -> None:
        """Pushes a message into the agent execution loop."""
        if hasattr(self.agent, "chat"):
            await self.agent.chat(message)


@dataclass
class Trigger:
    interval_seconds: float
    callback: Callable[[TriggerContext], Any]
    _task: Optional[asyncio.Task] = None

    async def start(self, ctx: TriggerContext) -> None:
        """Starts the periodic background execution loop."""
        async def _loop() -> None:
            while True:
                try:
                    await asyncio.sleep(self.interval_seconds)
                    res = self.callback(ctx)
                    if asyncio.iscoroutine(res):
                        await res
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error executing trigger callback: %s", str(e))

        self._task = asyncio.create_task(_loop())

    def stop(self) -> None:
        """Stops the trigger task execution."""
        if self._task and not self._task.done():
            self._task.cancel()


def every(seconds: float, callback: Callable[[TriggerContext], Any]) -> Trigger:
    """Helper function to create a periodic background trigger."""
    return Trigger(interval_seconds=seconds, callback=callback)


class TriggerRunner:
    """Manages active background triggers during agent session lifecycles."""

    def __init__(self, triggers: Optional[list[Trigger]] = None) -> None:
        self.triggers: list[Trigger] = triggers or []

    async def start_all(self, agent: Any) -> None:
        """Starts all registered background triggers."""
        ctx = TriggerContext(agent=agent)
        for trigger in self.triggers:
            await trigger.start(ctx)

    def stop_all(self) -> None:
        """Stops all running background triggers."""
        for trigger in self.triggers:
            trigger.stop()

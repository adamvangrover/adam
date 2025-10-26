# core/agents/reflector_agent.py
import logging
from typing import Any, Dict

from core.system.v22_async.async_agent_base import AsyncAgentBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReflectorAgent(AsyncAgentBase):
    """
    A simple agent that reflects its input back to the sender.
    Useful for testing agent routing and cyclical reasoning.
    """

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives a task and returns its payload.

        Args:
            task (Dict[str, Any]): The input task.

        Returns:
            Dict[str, Any]: The payload of the input task.
        """
        logger.info(f"Reflecting task: {task}")
        return task

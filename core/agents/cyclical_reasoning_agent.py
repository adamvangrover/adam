# core/agents/cyclical_reasoning_agent.py
import json
import logging
from typing import Any, Dict

from core.system.v22_async.async_agent_base import AsyncAgentBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CyclicalReasoningAgent(AsyncAgentBase):
    """
    An agent capable of cyclical reasoning, routing its output back to itself
    or other agents for iterative improvement.
    """

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a task, potentially routing it for further processing.

        Args:
            task (Dict[str, Any]): A dictionary containing the task details.
                Expected keys:
                - 'iterations_left' (int): The number of remaining iterations.
                - 'payload' (Any): The data to be processed.
                - 'target_agent' (str): The name of the next agent to process the payload.

        Returns:
            Dict[str, Any]: The final result after all iterations.
        """
        iterations_left = task.get('iterations_left', 0)
        payload = task.get('payload', {})
        target_agent = task.get('target_agent', self.name) # Default to self

        logger.info(f"Executing task. Iterations left: {iterations_left}. Payload: {payload}")

        if iterations_left <= 0:
            logger.info("No more iterations. Publishing final payload.")
            final_reply_to = task.get('final_reply_to')
            if final_reply_to:
                await self.send_message(final_reply_to, {"result": payload})
            return payload

        # Process the payload (for this simple agent, we just add a marker)
        payload['processed_by'] = self.name
        payload['iteration'] = payload.get('iteration', 0) + 1


        # Prepare the next task
        next_task = {
            'args': {
                'task': {
                    'iterations_left': iterations_left - 1,
                    'payload': payload,
                    'target_agent': task.get('next_target_agent', target_agent) # Agent for the *next* step
                }
            },
            'reply_to': task.get('final_reply_to') # Where the final result should go
        }

        logger.info(f"Routing task to {target_agent} with {iterations_left - 1} iterations left.")
        await self.send_message(target_agent, next_task)

        return None

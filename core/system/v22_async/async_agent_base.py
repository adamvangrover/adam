# core/system/v22_async/async_agent_base.py
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from semantic_kernel import Kernel
from core.system.message_broker import MessageBroker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AsyncAgentBase(ABC):
    """
    Abstract base class for asynchronous agents in the v22 system.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Kernel] = None):
        self.config = config
        self.constitution = constitution
        self.kernel = kernel
        self.message_broker = MessageBroker.get_instance()
        self.name = type(self).__name__
        self.context: Dict[str, Any] = {}
        log_message = f"Async Agent {self.name} initialized"
        if kernel:
            log_message += " with Semantic Kernel instance."
        logging.info(log_message)
        self.start_listening()

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

    def start_listening(self):
        """
        Subscribes the agent to its dedicated topic on the message broker.
        """
        self.message_broker.subscribe(self.name, self.handle_message)

    def handle_message(self, message: str):
        """
        Callback function to handle incoming messages.
        """
        data = json.loads(message)
        context = data.get("args")
        reply_to = data.get("reply_to")

        async def execute_and_reply():
            result = await self.execute(**context)
            if reply_to:
                reply_message = {"result": result}
                self.message_broker.publish(reply_to, json.dumps(reply_message))

        asyncio.create_task(execute_and_reply())

    async def send_message(self, target_agent: str, message: Dict[str, Any]):
        """
        Sends a message to another agent using the message broker.
        """
        self.message_broker.publish(target_agent, json.dumps(message))

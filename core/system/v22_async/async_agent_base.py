# core/system/v22_async/async_agent_base.py

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable

try:
    from semantic_kernel import Kernel
except ImportError:
    Kernel = Any # Fallback for type hinting
    logging.warning("Semantic Kernel not installed. AsyncAgentBase will run in reduced mode.")

from core.system.message_broker import MessageBroker
# Import Adaptive Heuristics
from core.system.v22_async.adaptive_heuristics import (
    ConvictionRouter,
    StateAnchorManager,
    ToolRegistry,
    SubscriptionManager
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AsyncAgentBase(ABC):
    """
    Abstract base class for asynchronous agents in the v22 system.
    Enhanced with Adaptive Conviction, State Anchors, and Subscription Patterns.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Any] = None):
        self.config = config
        self.constitution = constitution
        self.kernel = kernel
        self.message_broker = MessageBroker.get_instance()
        self.name = type(self).__name__
        self.context: Dict[str, Any] = {}

        # Initialize Adaptive System Components
        self.conviction_router = ConvictionRouter()
        self.state_manager = StateAnchorManager()
        self.subscription_manager = SubscriptionManager(self.message_broker)

        # Tool Registry initialization
        tools_list = self._extract_tools(kernel)
        self.tool_registry = ToolRegistry(tools_list)

        log_message = f"Async Agent {self.name} initialized"
        if kernel:
            log_message += " with Semantic Kernel instance."
        logging.info(log_message)
        self.start_listening()

    def _extract_tools(self, kernel) -> List[Dict[str, Any]]:
        """
        Helper to extract tool definitions from kernel or config.
        """
        tools = []
        if self.config.get("tools"):
            tools.extend(self.config["tools"])
        # If we had access to kernel plugins we could extract them here
        return tools

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

    async def adaptive_execute(self, task: str, **kwargs) -> Any:
        """
        Executes a task using Adaptive Conviction.
        Decides between Direct Execution (Low Conviction/Ambiguous) and Tool/MCP Execution (High Conviction).
        """
        # 1. Check Conviction
        conviction_high = self.conviction_router.should_use_mcp(task, self.context)

        if not conviction_high:
            # Low conviction: Use "Elicitation" or Direct Prompt
            logging.info(f"Low conviction for task '{task}'. Switching to Direct/Elicitation mode.")
            return {"status": "NEEDS_CLARIFICATION", "message": "Conviction too low. Please provide more details."}

        # 2. High Conviction: Use MCP/Tools
        # 2a. Check Context Budget (Tool RAG)
        relevant_tools = self.tool_registry.retrieve_tools(task)
        if relevant_tools:
            logging.info(f"Adaptive Execution: Using {len(relevant_tools)} relevant tools.")

        # 2b. State Anchor (if long running)
        is_long_running = kwargs.get("long_running", False) or "long_running" in task
        anchor_id = None
        if is_long_running:
            anchor_id = self.state_manager.create_anchor(task, self.context)

        # 3. Execute
        # We pass relevant tools to kwargs if the subclass supports it
        kwargs['relevant_tools'] = relevant_tools
        result = await self.execute(task=task, **kwargs)

        # 4. Verify Anchor (if created)
        if anchor_id:
            drift_safe = self.state_manager.verify_anchor(anchor_id, self.context)
            if not drift_safe:
                logging.warning("State Drift detected after execution!")
                if isinstance(result, dict):
                    result["warning"] = "STATE_DRIFT_DETECTED"

        return result

    def start_listening(self):
        """
        Subscribes the agent to its dedicated topic on the message broker.
        """
        self.message_broker.subscribe(self.name, self.handle_message)

    def subscribe_to_topic(self, topic: str, callback: Callable):
        """
        Subscribes to an arbitrary topic.
        """
        self.subscription_manager.subscribe(topic, callback)

    def publish_event(self, topic: str, event_type: str, data: Any):
        """
        Publishes an event with damping.
        """
        message = json.dumps({"type": event_type, "sender": self.name, "data": data})
        self.subscription_manager.publish_damped(topic, message)

    def handle_message(self, message: str):
        """
        Callback function to handle incoming messages.
        """
        try:
            data = json.loads(message)
            context = data.get("args", {})
            reply_to = data.get("reply_to")

            # Use adaptive execute if 'task' is present, else normal execute
            async def execute_and_reply():
                task = context.get("task")
                if task:
                    result = await self.adaptive_execute(task, **context)
                else:
                    result = await self.execute(**context)

                if reply_to:
                    reply_message = {"result": result}
                    self.message_broker.publish(reply_to, json.dumps(reply_message))

            asyncio.create_task(execute_and_reply())
        except Exception as e:
            logging.error(f"Error handling message: {e}")

    async def send_message(self, target_agent: str, message: Dict[str, Any]):
        """
        Sends a message to another agent using the message broker.
        """
        self.message_broker.publish(target_agent, json.dumps(message))

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import json
import asyncio
import uuid

# Import Kernel for type hinting
try:
    from semantic_kernel import Kernel
except ImportError:
    Kernel = Any  # Fallback for type hinting if package is missing


# Configure logging (you could also have a central logging config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AgentBase(ABC):
    """
    Abstract base class for all agents in the system.
    Defines the common interface and behavior expected of all agents.
    This version incorporates MCP, A2A, and Semantic Kernel.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Kernel] = None):
        """
        Initializes the AgentBase. Subclasses should call super().__init__(config, kernel)
        to ensure proper initialization. The config dictionary provides agent-specific
        configuration parameters, and kernel is an optional Semantic Kernel instance.
        """
        self.config = config
        self.constitution = constitution
        self.kernel = kernel # Store the Semantic Kernel instance
        self.context: Dict[str, Any] = {}
        self.peer_agents: Dict[str, AgentBase] = {}  # For A2A
        self.pending_requests: Dict[str, asyncio.Future] = {} # For Async A2A responses

        # Updated log message to reflect potential kernel presence
        log_message = f"Agent {type(self).__name__} initialized with config: {config}"
        if constitution:
            log_message += f" and constitution: {constitution.get('@id', 'N/A')}"
        if kernel:
            log_message += " and Semantic Kernel instance."
        else:
            log_message += "."
        logging.info(log_message)


    def set_context(self, context: Dict[str, Any]):
        """
        Sets the MCP context for the agent. This context contains
        information needed to perform the agent's task.
        """
        self.context = context
        logging.debug(f"Agent {type(self).__name__} context set: {context}")

    def get_context(self) -> Dict[str, Any]:
        """
        Returns the current MCP context.
        """
        return self.context

    def add_peer_agent(self, agent: 'AgentBase'):
        """
        Adds a peer agent for A2A communication.
        """
        self.peer_agents[agent.name] = agent
        logging.info(f"Agent {self.name} added peer agent: {agent.name}")

    async def send_message(self, target_agent: str, message: Dict[str, Any], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Sends an A2A message to another agent and waits for the response asynchronously.
        """
        correlation_id = str(uuid.uuid4())
        message['correlation_id'] = correlation_id
        # Ensure we have a return address. Ideally this is the queue name the agent listens on.
        message['reply_to'] = type(self).__name__

        future = asyncio.get_running_loop().create_future()
        self.pending_requests[correlation_id] = future

        try:
            if not hasattr(self, 'message_broker') or not self.message_broker:
                 logging.warning(f"Agent {type(self).__name__} has no message broker attached. Cannot send message.")
                 # Cleanup
                 del self.pending_requests[correlation_id]
                 return None

            self.message_broker.publish(target_agent, json.dumps(message))
            logging.info(f"Agent {type(self).__name__} sent message to {target_agent} (ID: {correlation_id})")

            return await asyncio.wait_for(future, timeout)

        except asyncio.TimeoutError:
            logging.error(f"Timeout waiting for response from {target_agent} (ID: {correlation_id})")
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
            return None
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]
            return None

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.
        This is the main entry point for agent execution.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

    def start_listening(self, message_broker):
        """
        Subscribes the agent to its dedicated topic on the message broker
        and processes incoming messages via a callback.
        """
        self.message_broker = message_broker # Store broker reference
        topic = type(self).__name__
        message_broker.subscribe(topic, self.handle_message)

    def handle_message(self, ch, method, properties, body):
        """
        Callback function to handle incoming messages.
        Dispatches to internal handler to manage async execution.
        """
        try:
            message = json.loads(body)
            correlation_id = message.get("correlation_id")

            # 1. Handle Response to a previous request
            if correlation_id and correlation_id in self.pending_requests:
                future = self.pending_requests.pop(correlation_id)
                if not future.done():
                    # pika callbacks might run in a different thread or context.
                    # We need to set the result in the loop where the future was created.
                    try:
                        loop = future.get_loop()
                        if not loop.is_closed():
                            loop.call_soon_threadsafe(future.set_result, message)
                        else:
                            logging.error("Event loop is closed, cannot set result for message.")
                    except Exception as loop_err:
                        logging.error(f"Error setting future result: {loop_err}")
                return

            # 2. Handle New Request (fire and forget / async)
            # Since 'execute' is async, we need to schedule it on an event loop.
            try:
                # Try to get the running loop if this callback is in the main thread
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # If we are in a separate thread (e.g. pika thread), we might not have a running loop accessible directly
                # In a real app, we should have a reference to the main loop.
                # For now, we attempt to get a loop or create a task if possible.
                # WARNING: Creating a new loop here is risky if one exists elsewhere.
                # Assuming standard asyncio usage where main loop handles this.
                logging.warning("No running loop found in handle_message. Request might be ignored if not properly scheduled.")
                return

            asyncio.run_coroutine_threadsafe(self._process_incoming_request(message), loop)

        except Exception as e:
            logging.error(f"Error in handle_message: {e}")

    async def _process_incoming_request(self, message: Dict[str, Any]):
        """
        Internal helper to process an incoming request asynchronously.
        """
        context = message.get("context", {})
        reply_to = message.get("reply_to")
        correlation_id = message.get("correlation_id")

        try:
            # Execute the agent's logic
            result = await self.execute(**context)

            # Publish the result to the reply_to topic if requested
            if reply_to and hasattr(self, 'message_broker'):
                response_message = result if isinstance(result, dict) else {"result": result}
                if correlation_id:
                    response_message["correlation_id"] = correlation_id

                self.message_broker.publish(reply_to, json.dumps(response_message))

        except Exception as e:
            logging.error(f"Error processing incoming request in agent {type(self).__name__}: {e}")
            # Optionally send error back
            if reply_to and hasattr(self, 'message_broker') and correlation_id:
                 error_msg = {"error": str(e), "correlation_id": correlation_id}
                 self.message_broker.publish(reply_to, json.dumps(error_msg))


    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the agent's skills (MCP). This should be overridden
        by subclasses to describe their specific capabilities.
        """
        return {
            "name": type(self).__name__,
            "description": self.config.get("description", "No description provided"),
            "skills": []
        }

    async def receive_message(self, sender_agent: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handles incoming A2A messages directly (alternative to broker).
        Subclasses should override this to define how they respond to messages.
        """
        logging.info(f"Agent {self.config.get('agent_id')} received message from {sender_agent}: {message}")
        return None  # Default: No response


    async def run_semantic_kernel_skill(self, skill_collection_name: str, skill_name: str, input_vars: Dict[str, str]) -> str:
        """
        Executes a Semantic Kernel skill. Standardized for clearer error handling.
        """
        if not hasattr(self, 'kernel') or not self.kernel:
            raise AttributeError("Agent does not have access to a Semantic Kernel instance.")

        try:
            # Support SK v1.x (Plugins)
            if hasattr(self.kernel, 'plugins') and skill_collection_name in self.kernel.plugins:
                plugin = self.kernel.plugins[skill_collection_name]
                if skill_name in plugin:
                    function = plugin[skill_name]
                    result = await self.kernel.invoke(function, **input_vars)
                    return str(result)

            # Support Older SK (Skills)
            if hasattr(self.kernel, 'skills') and hasattr(self.kernel.skills, 'get_function'):
                 function = self.kernel.skills.get_function(skill_collection_name, skill_name)
                 result = await self.kernel.run_async(function, input_vars=input_vars)
                 return str(result)

            # If we reach here, we couldn't find the skill
            raise ValueError(f"Skill '{skill_name}' in collection '{skill_collection_name}' not found in Kernel.")

        except Exception as e:
            logging.error(f"Error executing Semantic Kernel skill: {e}")
            raise

# Note: The 'Agent' class previously located here has been moved to 'core/agents/rag_agent.py'
# to improve architectural clarity and separation of concerns.

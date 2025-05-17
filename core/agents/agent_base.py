# core/agents/agent_base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import json
import asyncio

# Configure logging (you could also have a central logging config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AgentBase(ABC):
    """
    Abstract base class for all agents in the system.
    Defines the common interface and behavior expected of all agents.
    This version incorporates MCP and A2A.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AgentBase. Subclasses should call super().__init__(config)
        to ensure proper initialization. The config dictionary provides agent-specific
        configuration parameters.
        """
        self.config = config
        self.context: Dict[str, Any] = {}
        self.peer_agents: Dict[str, AgentBase] = {}  # For A2A
        logging.info(f"Agent {type(self).__name__} initialized with config: {config}")

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

    async def send_message(self, target_agent: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sends an A2A message to another agent and waits for the response.
        """
        if target_agent not in self.peer_agents:
            raise ValueError(f"Agent '{target_agent}' is not a known peer.")

        logging.info(f"Agent {self.name} sending message to {target_agent}: {message}")
        response = await self.peer_agents[target_agent].receive_message(self.name, message)
        logging.info(f"Agent {self.name} received response from {target_agent}: {response}")
        return response

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.
        This is the main entry point for agent execution.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

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
        Handles incoming A2A messages. Subclasses should override
        this to define how they respond to messages.
        """
        logging.info(f"Agent {self.name} received message from {sender_agent}: {message}")
        return None  # Default: No response


    async def run_semantic_kernel_skill(self, skill_name: str, input_vars: Dict[str, str]) -> str:
        """
        Executes a Semantic Kernel skill. This assumes the agent has access to
        a Semantic Kernel instance (self.kernel).
        """
        if not hasattr(self, 'kernel') or not self.kernel:
            raise AttributeError("Agent does not have access to a Semantic Kernel instance (self.kernel)")

        # Get the Semantic Kernel function for the given skill name.
        # This assumes skills are organized in a way that the skill name is also the function name.
        # You might need to adjust this based on how your skills are registered.
        sk_function = self.kernel.functions.get_function(skill_name)

        if not sk_function:
            raise ValueError(f"Semantic Kernel skill '{skill_name}' not found.")

        # Create the Semantic Kernel context.
        sk_context = self.kernel.create_new_context()

        # Set input variables for the Semantic Kernel function.
        for var_name, var_value in input_vars.items():
            sk_context[var_name] = var_value

        # Execute the Semantic Kernel function.
        result = await self.kernel.run(sk_function, input_vars=input_vars)

        # Return the result as a string.
        return str(result)

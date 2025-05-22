# core/agents/agent_base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional # Optional was already here, ensure Dict and Any are used consistently
import logging
import json
import asyncio

# Import Kernel for type hinting
from semantic_kernel import Kernel


# Configure logging (you could also have a central logging config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AgentBase(ABC):
    """
    Abstract base class for all agents in the system.
    Defines the common interface and behavior expected of all agents.
    This version incorporates MCP, A2A, and Semantic Kernel.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Kernel] = None):
        """
        Initializes the AgentBase. Subclasses should call super().__init__(config, kernel)
        to ensure proper initialization. The config dictionary provides agent-specific
        configuration parameters, and kernel is an optional Semantic Kernel instance.
        """
        self.config = config
        self.kernel = kernel # Store the Semantic Kernel instance
        self.context: Dict[str, Any] = {}
        self.peer_agents: Dict[str, AgentBase] = {}  # For A2A
        # Updated log message to reflect potential kernel presence
        log_message = f"Agent {type(self).__name__} initialized with config: {config}"
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


    async def run_semantic_kernel_skill(self, skill_collection_name: str, skill_name: str, input_vars: Dict[str, str]) -> str:
        """
        Executes a Semantic Kernel skill from a specific collection.
        This assumes the agent has access to a Semantic Kernel instance (self.kernel)
        and that skills have been imported into collections.
        """
        if not hasattr(self, 'kernel') or not self.kernel:
            raise AttributeError("Agent does not have access to a Semantic Kernel instance (self.kernel).")
        if not hasattr(self.kernel, 'skills') or not hasattr(self.kernel.skills, 'get_function'):
             raise AttributeError("Semantic Kernel instance does not have 'skills.get_function' method. SK version might be different than expected.")


        # Get the Semantic Kernel function from the specified collection and skill name.
        # For SK v1.x Python, this is typically kernel.plugins[plugin_name][function_name]
        # or kernel.skills.get_function(skill_collection_name, skill_name) if skills are registered that way.
        # The problem description suggests kernel.skills.get_function(skill_collection_name, skill_name).
        try:
            sk_function = self.kernel.skills.get_function(skill_collection_name, skill_name)
        except Exception as e: # Broad exception to catch issues if .skills or .get_function doesn't exist as expected
            logging.error(f"Error accessing SK function '{skill_name}' in collection '{skill_collection_name}': {e}. This might be due to an unexpected SK version or structure.")
            raise ValueError(f"Could not retrieve Semantic Kernel skill '{skill_name}' from collection '{skill_collection_name}'. Error: {e}")


        if not sk_function:
            raise ValueError(f"Semantic Kernel skill '{skill_name}' not found in collection '{skill_collection_name}'.")

        # Create the Semantic Kernel context.
        # For SK v1.x, context is often handled via KernelArguments or directly passed to invoke.
        # The method `kernel.create_new_context()` is from older versions (e.g., v0.9.x)
        # If using SK v1.x, `kernel.run_async(sk_function, input_vars=input_vars)` might not be the right way.
        # It would be more like: `await self.kernel.invoke(sk_function, **input_vars)`
        # or `await sk_function.invoke(variables=input_vars)`
        # Given the existing `self.kernel.run`, I will assume it's for an older SK version compatible with create_new_context.
        # However, the instruction for SK v1.x style get_function is a bit conflicting.
        # Let's stick to the existing run pattern but log a warning if create_new_context is missing.
        
        sk_context = None
        if hasattr(self.kernel, 'create_new_context') and callable(self.kernel.create_new_context):
            sk_context = self.kernel.create_new_context()
            # Set input variables for the Semantic Kernel function.
            for var_name, var_value in input_vars.items():
                sk_context[var_name] = var_value
        else:
            # If create_new_context is not available (e.g. SK v1.x), input_vars are usually passed directly to run/invoke.
            # The existing self.kernel.run call takes input_vars, so this might be fine.
            logging.debug("kernel.create_new_context() not found, assuming input_vars passed directly to kernel.run().")


        # Execute the Semantic Kernel function.
        # The existing code is: result = await self.kernel.run(sk_function, input_vars=input_vars)
        # For SK v1.x, it would be more like: result = await self.kernel.invoke(sk_function, **input_vars)
        # or await sk_function.invoke(input_vars)
        # Given the instruction to keep `kernel.run`, I'll use it.
        # If sk_context was created, some SK versions expect it in run: await self.kernel.run(sk_function, context=sk_context)
        # If not, input_vars directly: await self.kernel.run(sk_function, input_vars=input_vars)
        # The original code did not pass sk_context to run.
        
        # Using input_vars directly as per the original structure of kernel.run call
        result = await self.kernel.run_async(sk_function, input_vars=input_vars) # kernel.run is often run_async

        # Return the result as a string.

        # Set input variables for the Semantic Kernel function.
        for var_name, var_value in input_vars.items():
            sk_context[var_name] = var_value

        # Execute the Semantic Kernel function.
        result = await self.kernel.run(sk_function, input_vars=input_vars)

        # Return the result as a string.
        return str(result)

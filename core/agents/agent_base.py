# core/agents/agent_base.py
from abc import ABC, abstractmethod
from typing import Any
import logging

# Configure logging (you could also have a central logging config)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentBase(ABC):
    """
    Abstract base class for all agents in the system.
    Defines the common interface and behavior expected of all agents.
    """

    def __init__(self, **kwargs):
        """
        Initializes the AgentBase.  Subclasses should call super().__init__(**kwargs)
        to ensure proper initialization.  Any keyword arguments passed to the
        constructor will be set as attributes on the agent instance.
        """
        # Dynamically set attributes based on keyword arguments (for flexibility)
        for key, value in kwargs.items():
            setattr(self, key, value)
        logging.info(f"Agent {type(self).__name__} initialized.")


    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.
        This is the main entry point for agent execution.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The result of the agent's execution.  The type can vary
            depending on the agent's function.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")

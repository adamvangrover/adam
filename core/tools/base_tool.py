from abc import ABC, abstractmethod
from typing import Any

class BaseTool(ABC):
    """
    Abstract base class for tools that an agent can use.
    """
    name: str = "base_tool"
    description: str = "A base tool definition."

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Executes the tool's main functionality.
        """
        pass

    def get_schema(self) -> dict:
        """
        Returns a schema describing the tool, its purpose, and expected inputs/outputs.
        This can be used by planners (like Semantic Kernel) to understand how to use the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema(),
        }

    @abstractmethod
    def _get_parameters_schema(self) -> dict:
        """
        Helper to define the specific parameters for the tool's schema.
        """
        pass

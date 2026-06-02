"""
Purpose: Defines inter-module API boundaries and strict protocol definitions.
Dependencies: abc, typing
Outputs: EngineInterface Abstract Base Class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol


class EngineInterface(ABC):
    """
    Abstract base class for execution engines (e.g., Simulation, Live, Backtest).
    Forces strict contract boundaries across modules.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the engine with configuration parameters."""
        pass

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the engine logic given an input."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shut down engine operations."""
        pass

class SwarmOutput(Protocol):
    """Protocol defining the output boundary from the adam_swarm module."""
    data: Any
    confidence: float

class GraphInput(Protocol):
    """Protocol defining the input boundary into the adam_graph module."""
    nodes: Dict[str, Any]
    edges: list

class SwarmToGraphBoundary(ABC):
    """
    ABC for restricting adam_swarm output directly passing into adam_graph
    without interface transformation layers.
    """
    @abstractmethod
    def transform(self, swarm_output: SwarmOutput) -> GraphInput:
        """Transforms a SwarmOutput into a GraphInput."""
        pass

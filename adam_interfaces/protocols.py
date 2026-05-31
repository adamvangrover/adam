"""
Purpose: Defines inter-module API boundaries and strict protocol definitions.
Dependencies: abc, typing
Outputs: EngineInterface Abstract Base Class.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


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

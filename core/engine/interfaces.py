from abc import ABC, abstractmethod
from typing import Dict, Any

class EngineInterface(ABC):
    """
    Abstract base class for execution engines (e.g., Simulation, Live, Backtest).
    Enforces a standard contract for all market interaction layers.
    """

    @abstractmethod
    def get_market_pulse(self) -> Dict[str, Any]:
        """
        Retrieves the current market state.
        """
        pass

    @abstractmethod
    def generate_credit_memo(self, ticker: str, name: str = "Unknown", sector: str = "Unknown") -> Dict[str, Any]:
        """
        Generates a credit memo for a given entity.
        """
        pass

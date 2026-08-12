"""
Core Agnostic Math Module.
Provides universal, deterministic mathematical operations for all domains.
"""

from typing import List, Union
import statistics

class AgnosticMathEngine:
    @staticmethod
    def calculate_volatility(data_points: List[float]) -> float:
        """Calculate the standard deviation of a dataset."""
        if len(data_points) < 2:
            return 0.0
        return statistics.stdev(data_points)

    @staticmethod
    def calculate_var(data_points: List[float], confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) deterministically using historical simulation.
        Assumes data_points are returns.
        """
        if not data_points:
            return 0.0
        sorted_returns = sorted(data_points)
        index = int((1.0 - confidence_level) * len(sorted_returns))
        # Ensure we don't go out of bounds
        index = max(0, min(index, len(sorted_returns) - 1))
        return sorted_returns[index]

    @staticmethod
    def deterministic_hash(payload: Union[str, dict]) -> str:
        """Generate a deterministic hash for state freezing."""
        import hashlib
        import json
        if isinstance(payload, dict):
            payload_str = json.dumps(payload, sort_keys=True)
        else:
            payload_str = str(payload)
        return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()

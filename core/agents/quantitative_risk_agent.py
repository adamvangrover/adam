
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List
from core.agents.agent_base import AgentBase

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuantitativeRiskAgent(AgentBase):
    """
    Agent responsible for calculating quantitative risk metrics such as Value at Risk (VaR)
    and Conditional Value at Risk (CVaR).
    """
    def __init__(self, config: Dict[str, Any] = None, constitution: Dict[str, Any] = None, kernel: Any = None):
        super().__init__(config, constitution, kernel)
        self.confidence_level = self.config.get('confidence_level', 0.95)

    async def execute(self, *args, **kwargs):
        """
        Calculates risk metrics for the provided portfolio returns.
        Expected kwargs:
            - returns: list or pd.Series of portfolio returns
        """
        returns_data = kwargs.get('returns')

        if returns_data is None:
            # Try args[0] if it looks like data
            if args and isinstance(args[0], (list, np.ndarray, pd.Series)):
                returns_data = args[0]
            else:
                return {"status": "error", "message": "No returns data provided."}

        try:
            if isinstance(returns_data, list):
                returns = np.array(returns_data)
            elif isinstance(returns_data, (pd.Series, pd.DataFrame)):
                returns = returns_data.values
            elif isinstance(returns_data, np.ndarray):
                returns = returns_data
            else:
                return {"status": "error", "message": f"Invalid data format: {type(returns_data)}"}

            var = self.calculate_var(returns)
            cvar = self.calculate_cvar(returns)

            return {
                "status": "success",
                "metrics": {
                    "VaR": var,
                    "CVaR": cvar,
                    "confidence_level": self.confidence_level
                }
            }
        except Exception as e:
            logging.error(f"Error in QuantitativeRiskAgent: {e}")
            return {"status": "error", "message": str(e)}

    def calculate_var(self, returns: np.ndarray) -> float:
        """Calculates Value at Risk (VaR)"""
        if not HAS_SCIPY:
            # Historical simulation fallback
            return float(np.percentile(returns, (1 - self.confidence_level) * 100))

        mu = np.mean(returns)
        sigma = np.std(returns)
        return float(norm.ppf(1 - self.confidence_level, mu, sigma))

    def calculate_cvar(self, returns: np.ndarray) -> float:
        """Calculates Conditional Value at Risk (CVaR)"""
        var = self.calculate_var(returns)
        return float(returns[returns <= var].mean())

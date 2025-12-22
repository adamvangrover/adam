import logging
import random
from typing import Dict, Any, List

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class WealthManager:
    """
    Manages Wealth Planning, Trusts, and Beneficiary Goals.
    Includes Monte Carlo simulation for goal probability.
    """

    def plan_goal(self, goal_name: str, target_amount: float, horizon_years: int, current_savings: float) -> Dict[str, Any]:
        """
        Creates a funding plan for a specific goal (e.g., Philanthropy, NextGen Education).
        Uses Monte Carlo simulation to estimate probability of success.
        """
        logger.info(f"Planning goal: {goal_name}")

        # Assumptions
        mean_return = 0.07
        std_dev = 0.12
        simulations = 1000

        prob_success = 0.0
        median_terminal_value = 0.0

        if NUMPY_AVAILABLE:
            # Vectorized MC
            # Returns matrix: (simulations, horizon)
            returns = np.random.normal(mean_return, std_dev, (simulations, horizon_years))
            # Cumulative growth factors: (simulations, horizon) -> prod along axis 1 -> (simulations,)
            growth_factors = np.prod(1 + returns, axis=1)
            terminal_values = current_savings * growth_factors

            successes = np.sum(terminal_values >= target_amount)
            prob_success = successes / simulations
            median_terminal_value = np.median(terminal_values)
        else:
            # Fallback Loop MC
            success_count = 0
            terminal_values = []
            for _ in range(simulations):
                val = current_savings
                for _ in range(horizon_years):
                    r = random.gauss(mean_return, std_dev)
                    val *= (1 + r)
                terminal_values.append(val)
                if val >= target_amount:
                    success_count += 1

            prob_success = success_count / simulations
            median_terminal_value = sorted(terminal_values)[simulations // 2]

        shortfall = target_amount - median_terminal_value
        status = "On Track" if prob_success > 0.75 else ("At Risk" if prob_success > 0.5 else "Critical")

        return {
            "goal": goal_name,
            "horizon": horizon_years,
            "current_savings": current_savings,
            "target": target_amount,
            "projected_median_value": round(float(median_terminal_value), 2),
            "probability_of_success": round(float(prob_success), 2),
            "shortfall": round(max(0, float(shortfall)), 2),
            "status": status,
            "recommendation": self._get_recommendation(status)
        }

    def _get_recommendation(self, status: str) -> str:
        if status == "On Track":
            return "Maintain current strategy. Consider harvesting gains."
        elif status == "At Risk":
            return "Consider increasing savings rate or allocation to Growth Assets."
        else:
            return "Urgent: Revise goal target or inject significant capital."

    def structure_trust(self, trust_name: str, beneficiaries: List[str]) -> Dict[str, Any]:
        """
        Models a trust structure.
        """
        return {
            "name": trust_name,
            "type": "Irrevocable Grantor Trust",
            "beneficiaries": beneficiaries,
            "tax_status": "Pass-Through",
            "assets": []
        }

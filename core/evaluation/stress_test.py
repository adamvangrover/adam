from typing import Dict, Any, List, Optional, Callable
import copy
import logging
from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine

logger = logging.getLogger(__name__)

class AdversarialSimulator:
    """
    Layer 3: Stress-Testing (Adversarial).
    Builds a 'Red Team' to inject zombie data and perform sensitivity analysis.
    Now utilizes GenerativeRiskEngine for advanced scenario generation.
    """

    def __init__(self):
        # Initialize generative engine for advanced scenarios
        try:
            self.gen_engine = GenerativeRiskEngine()
        except Exception:
            self.gen_engine = None

    def zombie_attack(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intentionally injects 'zombie company' data (high debt, low interest coverage)
        to see if the risk-assessment agent catches it.

        Zombie Definition (BIS):
        - Interest Coverage Ratio (ICR) < 1
        - Low growth expectations
        """
        attack_data = copy.deepcopy(financial_data)

        # Method 1: Generative Crash Scenario (If available)
        if self.gen_engine:
            try:
                # Generate a "Crash" regime scenario
                scenarios = self.gen_engine.generate_scenarios(n_samples=1, regime="crash")
                crash_scenario = scenarios[0]

                # Apply Macro factors to company data
                # High inflation -> higher interest expense
                # Low GDP -> lower EBITDA
                inf = crash_scenario.risk_factors.get("inflation", 2.0)
                gdp = crash_scenario.risk_factors.get("gdp_growth", 2.0)

                interest_multiplier = 1.0 + (max(0, inf - 2.0) * 0.1) # +10% expense per 1% excess inflation
                revenue_multiplier = 1.0 + (min(0, gdp) * 0.05) # -5% revenue per -1% GDP drop

                attack_data["interest_expense"] = attack_data.get("interest_expense", 100000) * interest_multiplier
                current_ebitda = attack_data.get("ebitda", 1000000)
                attack_data["ebitda"] = current_ebitda * revenue_multiplier

                logger.info(f"Injecting Generative Crash Scenario: Inf={inf:.1f}%, GDP={gdp:.1f}%")

                # If still not a zombie (ICR > 1), force it
                if attack_data["ebitda"] > attack_data["interest_expense"]:
                     # Force zombie logic as backup
                     attack_data["ebitda"] = attack_data["interest_expense"] * 0.8

                return attack_data
            except Exception as e:
                logger.warning(f"Generative engine failed: {e}. Falling back to heuristic.")

        # Method 2: Heuristic Fallback
        # Assume standard keys, or defaulting if missing
        interest_expense = attack_data.get("interest_expense", 1000000)

        # 1. Set EBITDA below Interest Expense (ICR < 1.0)
        # Setting ICR to 0.8
        new_ebitda = interest_expense * 0.8
        attack_data["ebitda"] = new_ebitda

        # 2. Inflate Total Debt to increase leverage (Debt/EBITDA > 6x)
        attack_data["total_debt"] = new_ebitda * 8.0

        # 3. Reduce Cash to create liquidity crunch
        attack_data["cash_and_equivalents"] = interest_expense * 0.2

        logger.info("Injecting Zombie Data (Heuristic): EBITDA reduced, Debt inflated.")
        return attack_data

    def sensitivity_probe(self,
                          agent_func: Callable[[Dict[str, Any]], float],
                          base_data: Dict[str, Any],
                          param_name: str,
                          start: float,
                          end: float,
                          steps: int) -> List[Dict[str, float]]:
        """
        Automates a workflow that slightly tweaks a parameter (e.g., Interest Rates, EBITDA)
        and checks if the AI's risk rating changes proportionally.

        Args:
            agent_func: A wrapper function that takes data and returns a risk score (0-100 or 0.0-1.0).
            base_data: The initial dataset.
            param_name: The key to modify in base_data.
            start: Start value.
            end: End value.
            steps: Number of steps.
        """
        results = []
        step_size = (end - start) / (steps - 1) if steps > 1 else 0

        current_val = start
        for _ in range(steps):
            # Create test payload
            test_data = copy.deepcopy(base_data)
            test_data[param_name] = current_val

            # Execute Agent
            try:
                risk_score = agent_func(test_data)
            except Exception as e:
                logger.error(f"Agent failed during sensitivity probe: {e}")
                risk_score = -1.0

            results.append({
                param_name: current_val,
                "risk_score": risk_score
            })

            current_val += step_size

        return results

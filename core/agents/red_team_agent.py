from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import random
from core.agents.agent_base import AgentBase

# Try to import the v23 Graph. If it fails (e.g. missing deps), we fall back to mock logic.
try:
    from core.v23_graph_engine.red_team_graph import red_team_app
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    red_team_app = None

class RedTeamAgent(AgentBase):
    """
    The Red Team Agent acts as an adversary to the system.
    It generates novel and challenging scenarios (stress tests) to validate risk models.
    In v23, it wraps the `RedTeamGraph` for cyclical adversarial generation.
    """

    def __init__(self, config: Dict[str, Any], kernel=None):
        super().__init__(config, kernel=kernel)
        self.scenarios = [
            "Sudden interest rate hike",
            "Global pandemic recurrence",
            "Supply chain collapse in semiconductor sector",
            "Major cybersecurity breach at top 5 bank",
            "Geopolitical conflict in key trade region",
            "Currency hyperinflation in emerging market",
            "Regulatory crackdown on AI trading"
        ]

    async def execute(self, target_portfolio: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generates adversarial scenarios to test the system or a specific portfolio.
        """
        logging.info("RedTeamAgent: Generating adversarial scenarios...")

        target_name = "General Portfolio"
        if target_portfolio:
            target_name = target_portfolio.get("id", str(target_portfolio))

        # v23 Path: Use the Graph Engine
        if GRAPH_AVAILABLE and red_team_app:
            logging.info("RedTeamAgent: delegating to v23 RedTeamGraph.")
            initial_state = {
                "target_entity": target_name,
                "scenario_type": "Macro", # Default, could be parameterized
                "current_scenario_description": "",
                "simulated_impact_score": 0.0,
                "severity_threshold": 7.5,
                "critique_notes": [],
                "iteration_count": 0,
                "is_sufficiently_severe": False,
                "human_readable_status": "Agent initiating Red Team Graph..."
            }

            try:
                # Use ainvoke for async execution
                result = await red_team_app.ainvoke(initial_state)
                return {
                    "graph_output": result,
                    "summary": f"Red Team Graph finished with status: {result.get('human_readable_status')}"
                }
            except Exception as e:
                logging.error(f"RedTeamGraph execution failed: {e}. Falling back to legacy logic.")
                # Fallthrough to legacy logic

        # v21 Path: Legacy Random Selection
        logging.info("RedTeamAgent: Using legacy random scenario generation.")
        num_scenarios = self.config.get("num_scenarios", 2)
        selected_scenarios = random.sample(self.scenarios, k=min(num_scenarios, len(self.scenarios)))

        generated_stress_test = {
            "scenarios": selected_scenarios,
            "severity": "High",
            "target_portfolio_id": target_name,
            "description": "Adversarial conditions generated to test resilience (Legacy Mode)."
        }

        logging.info(f"RedTeamAgent: Generated scenarios: {selected_scenarios}")

        return generated_stress_test

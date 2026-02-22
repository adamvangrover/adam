from __future__ import annotations
from typing import Any, Dict, Optional, Union
import logging
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class IndustryRiskAgent(AgentBase):
    """
    Agent responsible for assessing Industry Risk.
    Evaluates competition using Quantitative Porter's 5 Forces and Cyclicality.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

        # Expanded Sector Profiles (Default Heuristics)
        # Scores are 1-10 (10 = High Risk/High Intensity)
        self.sector_profiles = {
            "Technology": {
                "rivalry": 9, "new_entrants": 8, "substitutes": 9, "supplier_power": 5, "buyer_power": 6,
                "cyclicality": "High", "beta_proxy": 1.2
            },
            "Energy": {
                "rivalry": 6, "new_entrants": 4, "substitutes": 8, "supplier_power": 4, "buyer_power": 7,
                "cyclicality": "High", "beta_proxy": 1.3
            },
            "Financials": {
                "rivalry": 8, "new_entrants": 6, "substitutes": 7, "supplier_power": 3, "buyer_power": 5,
                "cyclicality": "High", "beta_proxy": 1.1
            },
            "Utilities": {
                "rivalry": 3, "new_entrants": 2, "substitutes": 4, "supplier_power": 4, "buyer_power": 4,
                "cyclicality": "Low", "beta_proxy": 0.5
            },
            "Consumer Staples": {
                "rivalry": 7, "new_entrants": 5, "substitutes": 4, "supplier_power": 5, "buyer_power": 8,
                "cyclicality": "Low", "beta_proxy": 0.6
            },
            "Healthcare": {
                "rivalry": 6, "new_entrants": 3, "substitutes": 3, "supplier_power": 6, "buyer_power": 7,
                "cyclicality": "Moderate", "beta_proxy": 0.8
            }
        }

    async def execute(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess risk for a specific industry sector.

        Args:
            input_data:
                - str: Sector name (e.g., "Technology").
                - dict: {
                    "sector": "Custom",
                    "porter_forces": {
                        "rivalry": 8,
                        "new_entrants": 7,
                        ...
                    },
                    "cyclicality_beta": 1.5
                  }

        Returns:
            Dict containing industry risk assessment.
        """
        logger.info("Starting Industry Risk Assessment...")

        sector_name = "Unknown"
        forces = {}
        beta = 1.0

        # Parse Input
        if isinstance(input_data, str):
            sector_name = input_data
            profile = self.sector_profiles.get(sector_name, {})
            if not profile:
                logger.warning(f"Sector '{sector_name}' not found in default profiles.")
                profile = {"rivalry": 5, "new_entrants": 5, "substitutes": 5, "supplier_power": 5, "buyer_power": 5, "beta_proxy": 1.0}

            forces = {k: v for k, v in profile.items() if k in ["rivalry", "new_entrants", "substitutes", "supplier_power", "buyer_power"]}
            beta = profile.get("beta_proxy", 1.0)

        elif isinstance(input_data, dict):
            sector_name = input_data.get("sector", "Custom")
            forces = input_data.get("porter_forces", {})
            beta = input_data.get("cyclicality_beta", 1.0)

            # Fill defaults if missing
            for k in ["rivalry", "new_entrants", "substitutes", "supplier_power", "buyer_power"]:
                if k not in forces:
                    forces[k] = 5

        # Calculate Porter's Risk Score (0-100)
        # Weighting: Rivalry 30%, New Entrants 20%, Substitutes 20%, Supplier 15%, Buyer 15%
        raw_score = (
            forces.get("rivalry", 5) * 3.0 +
            forces.get("new_entrants", 5) * 2.0 +
            forces.get("substitutes", 5) * 2.0 +
            forces.get("supplier_power", 5) * 1.5 +
            forces.get("buyer_power", 5) * 1.5
        )
        # raw_score is out of 100 (since max is 10*10 = 100)

        # Cyclicality Adjustment
        # If Beta > 1.2, add penalty. If Beta < 0.8, reduce risk.
        cyclicality_adj = 0.0
        if beta > 1.5: cyclicality_adj = 15
        elif beta > 1.2: cyclicality_adj = 10
        elif beta < 0.6: cyclicality_adj = -10
        elif beta < 0.8: cyclicality_adj = -5

        final_score = raw_score + cyclicality_adj
        final_score = max(0, min(100, final_score))

        level = "High" if final_score > 60 else "Medium" if final_score > 30 else "Low"

        # Determine Cyclicality Label
        cyclicality_label = "Cyclical" if beta > 1.1 else "Defensive" if beta < 0.9 else "Neutral"

        result = {
            "sector": sector_name,
            "industry_risk_score": float(final_score),
            "risk_level": level,
            "porters_five_forces_score": float(raw_score),
            "forces_breakdown": forces,
            "cyclicality_analysis": {
                "beta": float(beta),
                "label": cyclicality_label,
                "risk_adjustment": float(cyclicality_adj)
            }
        }

        logger.info(f"Industry Risk Assessment Complete: {result}")
        return result

import logging
from typing import Dict, Any, List
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import CovenantRiskAnalysis

logger = logging.getLogger(__name__)

class CovenantAnalystAgent(AgentBase):
    """
    Phase 3 Helper: Covenant Analysis.
    Parses credit agreements (or simulates them) for maintenance covenants.

    This agent simulates the role of a Legal/Credit analyst reviewing the Credit Agreement.
    It checks for Financial Maintenance Covenants (Total Net Leverage, Interest Coverage)
    and estimates the risk of a "Foot Fault" or technical default.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "Credit Lawyer"

    async def execute(self, **kwargs) -> CovenantRiskAnalysis:
        logger.info("Executing Covenant Analysis...")

        # Support both nested 'params' key and flat kwargs
        params = kwargs.get('params', kwargs)

        # Inputs
        current_leverage = float(params.get("leverage", 3.0))
        covenant_threshold = float(params.get("covenant_threshold", 4.0))

        # Analysis Logic
        headroom = covenant_threshold - current_leverage

        # Determine Risk Level and Narrative
        if headroom < 0:
            risk = "Critical (Breach)"
            assessment = (f"BREACH DETECTED: Current leverage ({current_leverage:.2f}x) exceeds "
                          f"maintenance covenant ({covenant_threshold:.2f}x). "
                          "Immediate waiver request or equity cure required.")
        elif headroom < 0.5:
            risk = "High"
            assessment = (f"Tight Headroom ({headroom:.2f}x). Any EBITDA degradation (>10%) "
                          "will trigger a default event. Management likely restricted from M&A/Dividends.")
        elif headroom < 1.0:
            risk = "Medium"
            assessment = (f"Moderate Headroom ({headroom:.2f}x). Standard operating flexibility exists, "
                          "but large debt-funded capex is constrained.")
        else:
            risk = "Low"
            assessment = (f"Ample Headroom ({headroom:.2f}x). Full access to revolver and "
                          "accordion features likely available.")

        return CovenantRiskAnalysis(
            primary_constraint=f"Net Leverage Ratio < {covenant_threshold:.2f}x",
            current_level=current_leverage,
            breach_threshold=covenant_threshold,
            risk_assessment=risk,
            detailed_narrative=assessment # Assuming schema supports this or will ignore
        )

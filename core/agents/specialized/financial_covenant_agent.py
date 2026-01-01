import logging
from typing import Dict, Any, List
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import CovenantRiskAnalysis

logger = logging.getLogger(__name__)


class CovenantAnalystAgent(AgentBase):
    """
    Phase 3 Helper: Covenant Analysis.
    Parses credit agreements (or simulates them) for maintenance covenants.

    Enhanced Capabilities:
    - Technical Default Prediction (Headroom Compression)
    - Springing Covenant Monitoring (Revolver Utilization)
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

        # New Inputs for Enhanced Analysis
        prev_leverage = float(params.get("prev_leverage", current_leverage)) # For trend analysis
        revolver_drawn = float(params.get("revolver_drawn", 0.0))
        revolver_cap = float(params.get("revolver_cap", 100.0))
        springing_trigger_pct = float(params.get("springing_trigger_pct", 0.30)) # Standard 30% utilization trigger

        # 1. Standard Maintenance Covenant Analysis
        headroom = covenant_threshold - current_leverage
        headroom_pct = (headroom / covenant_threshold) * 100 if covenant_threshold > 0 else 0.0

        # 2. Trend Analysis (Headroom Compression)
        leverage_delta = current_leverage - prev_leverage
        compression_warning = ""
        if leverage_delta > 0.5:
            compression_warning = f" | WARNING: Rapid compression (Lev +{leverage_delta:.1f}x QoQ)."

        # 3. Springing Covenant Analysis
        utilization = revolver_drawn / revolver_cap if revolver_cap > 0 else 0.0
        springing_risk = ""
        if utilization > springing_trigger_pct:
            springing_risk = f" | SPRINGING COVENANT ACTIVE (Util: {utilization:.1%} > {springing_trigger_pct:.1%})."
        elif utilization > (springing_trigger_pct - 0.10):
            springing_risk = f" | Springing Test Approaching (Util: {utilization:.1%})."

        # 4. Determine Risk Level and Narrative
        risk_narrative = ""
        if headroom < 0:
            risk = "Critical (Breach)"
            risk_narrative = (f"BREACH DETECTED: Current leverage ({current_leverage:.2f}x) exceeds "
                          f"maintenance covenant ({covenant_threshold:.2f}x). "
                          "Immediate waiver request or equity cure required.")
        elif headroom < 0.5:
            risk = "High"
            risk_narrative = (f"Tight Headroom ({headroom:.2f}x / {headroom_pct:.1f}%). Any EBITDA degradation (>10%) "
                          "will trigger a default event. Management likely restricted from M&A/Dividends.")
        elif headroom < 1.0:
            risk = "Medium"
            risk_narrative = (f"Moderate Headroom ({headroom:.2f}x / {headroom_pct:.1f}%). Standard operating flexibility exists, "
                          "but large debt-funded capex is constrained.")
        else:
            risk = "Low"
            risk_narrative = (f"Ample Headroom ({headroom:.2f}x / {headroom_pct:.1f}%). Full access to revolver and "
                          "accordion features likely available.")

        # Combine narratives
        full_assessment = risk_narrative + compression_warning + springing_risk

        return CovenantRiskAnalysis(
            primary_constraint=f"Net Leverage Ratio < {covenant_threshold:.2f}x",
            current_level=current_leverage,
            breach_threshold=covenant_threshold,
            headroom_assessment=full_assessment
        )

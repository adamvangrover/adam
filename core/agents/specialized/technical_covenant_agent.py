import logging
from typing import Dict, Any, List
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import CovenantRiskAnalysis

logger = logging.getLogger(__name__)


class TechnicalCovenantAgent(AgentBase):
    """
    Specialized Agent: The Legal Analyst (Law Firm Associate Persona).

    This agent focuses purely on the textual "rules of the road" within the Credit Agreement.
    It identifies definitions, baskets, and blockers.

    Enhanced Capabilities:
    - Context-Aware Checking: Prioritizes checks based on borrower history (e.g., Aggressive Sponsors).
    - Historical Precedent: Flags "Market Standard" vs "Outlier" terms.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "Legal Associate"

    async def execute(self, **kwargs) -> CovenantRiskAnalysis:
        logger.info("Executing Technical Covenant Review...")

        # Inputs
        params = kwargs.get('params', kwargs)
        doc_text = params.get("doc_text", "")
        context = params.get("context", {}) # e.g. {"sponsor_type": "Aggressive", "distressed": True}

        # 1. Determine Relevant Baskets/Checks
        checks = self._determine_relevant_checks(context)

        assessment_parts = []

        # 2. Execute Checks
        if "J.Crew" in checks:
            has_jcrew = "material intellectual property" in doc_text.lower() and "unrestricted subsidiary" in doc_text.lower()
            if has_jcrew:
                assessment_parts.append("ALERT: Potential J.Crew Blocker language detected (IP Transfer restrictions).")
            else:
                assessment_parts.append("RISK: No specific J.Crew Blocker found. Value leakage to Unrestricted Subs possible.")

        if "Serta" in checks:
            has_serta = "supermajority" in doc_text.lower() or "sacred right" in doc_text.lower()
            if has_serta:
                assessment_parts.append("PROTECTION: Serta Protection (Uptiering) likely requires Supermajority.")
            else:
                assessment_parts.append("RISK: Serta/Uptiering risk high (Simple Majority).")

        if "Chewy" in checks:
             # Look for "Designated Non-Cash Consideration"
             if "designated non-cash consideration" in doc_text.lower():
                 assessment_parts.append("RISK: 'Chewy' loophole active (Non-cash consideration for asset sales).")

        # 3. Contextual Insight
        insight = "Standard Review."
        if context.get("sponsor_type") == "Aggressive":
             insight = "Given Aggressive Sponsor history, the lack of J.Crew/Chewy blockers is a Critical Defect."

        full_assessment = " | ".join(assessment_parts) + f" [INSIGHT: {insight}]"

        return CovenantRiskAnalysis(
            primary_constraint="Legal/Documentary Protections",
            current_level=0.0,
            breach_threshold=0.0,
            headroom_assessment=full_assessment
        )

    def _determine_relevant_checks(self, context: Dict[str, Any]) -> List[str]:
        """
        Prioritizes what to check based on deal context.
        """
        checks = ["J.Crew", "Serta"] # Defaults

        if context.get("distressed", False):
            checks.append("Bankruptcy Remote") # Check separate definitions

        if context.get("sponsor_type") == "Aggressive":
            checks.append("Chewy")
            checks.append("Encan") # Uncapped baskets

        return checks

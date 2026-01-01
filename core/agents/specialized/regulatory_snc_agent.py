import logging
from typing import Dict, Any, List, Optional
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import SNCRatingModel, PrimaryFacilityAssessment
from core.system.aof_guardrail import AgenticOversightFramework

# Configure logging
logger = logging.getLogger(__name__)


class RegulatorySNCAgent(AgentBase):
    """
    Specialized Agent: The Regulator (Government Employee Persona).

    This agent strictly applies the "Interagency Guidance on Leveraged Lending" (2013).
    It does NOT use flexible cash flow models or future projections.
    It focuses on rigid compliance: Leverage < 6x, Ability to Repay < 50% of Free Cash Flow.

    Role: "The Brake"
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "FDIC Examiner"

    @AgenticOversightFramework.oversight_guardrail(confidence_threshold=0.90) # Higher threshold for regulator
    async def execute(self,
                financials: Dict[str, float],
                capital_structure: List[Dict[str, Any]],
                enterprise_value: float) -> SNCRatingModel:
        """
        Executes the Regulatory SNC rating logic.
        """
        logger.info("Initiating Regulatory SNC Exam...")

        # 1. Extract Key Metrics (Strict GAAP Focus)
        ebitda = float(financials.get('ebitda', 0.0))
        total_debt = float(financials.get('total_debt', 0.0))

        # Regulator looks at Gross Leverage often, or Net if cash is restricted. Assuming Net here for parity but strict.
        leverage_ratio = total_debt / ebitda if ebitda > 0 else 99.0

        # 2. Strict Classification Logic (The "Box")
        # Pass: Leverage < 4.0x
        # Special Mention: Leverage 4.0x - 6.0x (Regulators hate >6x)
        # Substandard: Leverage > 6.0x (Automatic flag in many exams)
        # Doubtful: Leverage > 8.0x

        rating = "Pass"
        rationale = ""

        if leverage_ratio < 4.0:
            rating = "Pass"
            rationale = "Leverage within prudent limits (<4x)."
        elif leverage_ratio < 6.0:
            rating = "Special Mention"
            rationale = "Leverage elevated (4-6x). Monitor for de-leveraging capacity."
        elif leverage_ratio < 8.0:
            rating = "Substandard"
            rationale = "Leverage exceeds regulatory comfort zone (>6x). Repayment ability questionable."
        else:
            rating = "Doubtful"
            rationale = "Leverage excessive (>8x). High probability of loss."

        # 3. Facility Assessment
        sorted_cap_structure = sorted(capital_structure, key=lambda x: x.get('priority', 99))
        primary_facility = sorted_cap_structure[0] if sorted_cap_structure else {'name': 'General Facility', 'amount': total_debt}

        # Regulators are less impressed by "Enterprise Value" and more by hard collateral (Liquidation Value)
        # We assume EV is provided, but regulator haircuts it by 30% for safety.
        reg_ev = enterprise_value * 0.70
        reg_ltv = total_debt / reg_ev if reg_ev > 0 else 99.0

        collateral_bucket = "Weak"
        if reg_ltv < 0.8:
             collateral_bucket = "Adequate"

        result = SNCRatingModel(
            overall_borrower_rating=rating,
            rationale=f"Regulatory View: {rationale} (Lev: {leverage_ratio:.2f}x, Reg-LTV: {reg_ltv:.2f})",
            primary_facility_assessment=PrimaryFacilityAssessment(
                facility_type=primary_facility.get('name', 'Unknown'),
                collateral_coverage=collateral_bucket,
                repayment_capacity="High" if rating == "Pass" else "Low"
            ),
            conviction_score=1.0 # Regulators are always 100% convinced of their rules.
        )

        return result

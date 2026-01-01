import logging
import asyncio
from typing import Dict, Any, List, Optional
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import SNCRatingModel, PrimaryFacilityAssessment
from core.system.aof_guardrail import AgenticOversightFramework
from core.engine.risk_consensus_engine import RiskConsensusEngine
from core.agents.specialized.regulatory_snc_agent import RegulatorySNCAgent

# Configure logging
logger = logging.getLogger(__name__)


class StrategicSNCAgent(AgentBase):
    """
    Specialized Agent for performing Shared National Credit (SNC) simulations.

    Acts as a virtual 'Senior Credit Officer', orchestrating the debate between:
    1. The Regulator (RegulatorySNCAgent) - "The Brake"
    2. The Strategist (Internal Logic) - "The Gas"

    It uses the Risk Consensus Engine to simulate a dialogue and determine the final outcome.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "Bank Risk Officer"
        # Initialize engines and sub-agents
        self.consensus_engine = RiskConsensusEngine()
        # In a real system, this might be injected, but instantiating for direct dependency here
        self.regulator = RegulatorySNCAgent(config)

    @AgenticOversightFramework.oversight_guardrail(confidence_threshold=0.85)
    async def execute(self,
                financials: Dict[str, float],
                capital_structure: List[Dict[str, Any]],
                enterprise_value: float) -> SNCRatingModel:
        """
        Executes the Dual-Model Consensus Process.
        """
        logger.info("Initiating Risk Consensus Process...")

        # 1. Get Regulatory View (The Brake)
        # We explicitly run the Regulatory Agent to get the compliance baseline
        reg_result = await self.regulator.execute(financials, capital_structure, enterprise_value)
        reg_rating = reg_result.overall_borrower_rating
        reg_rationale = reg_result.rationale

        # 2. Generate Strategic View (The Gas)
        # Internal Logic (DSCR/LTV - 2025 Model)
        ebitda = float(financials.get('ebitda', 0.0))
        total_debt = float(financials.get('total_debt', 0.0))
        interest_expense = float(financials.get('interest_expense', 1.0))
        if interest_expense <= 0: interest_expense = 1.0

        capex = float(financials.get('capex', 0.0))
        principal = float(financials.get('principal_payment', 0.0))

        numerator = ebitda - capex
        denominator = interest_expense + principal
        dscr = numerator / denominator if denominator > 0 else (numerator / interest_expense if interest_expense > 0 else 0.0)
        ltv = total_debt / enterprise_value if enterprise_value > 0 else 99.0
        collateral_quality = "Medium"

        if dscr >= 1.2 and ltv <= 0.8:
            new_rating = "Pass"
            new_score = 0.95
            strat_rationale = f"Strong repayment capacity (DSCR {dscr:.2f}x) supports Pass rating despite leverage."
        elif dscr < 0.8 and collateral_quality == 'Low':
            new_rating = "Doubtful"
            new_score = 0.99
            strat_rationale = "Cash flow insufficient to service debt."
        elif dscr < 1.0 or ltv > 0.9:
            new_rating = "Substandard"
            new_score = 0.90
            strat_rationale = "Repayment relies on collateral liquidation."
        elif dscr < 1.2 or ltv > 0.8:
            new_rating = "Special Mention"
            new_score = 0.85
            strat_rationale = "Tight coverage requires monitoring."
        else:
            new_rating = "Special Mention"
            new_score = 0.70
            strat_rationale = "Metrics borderline."

        # 3. Consensus & Dialogue
        consensus_result = self.consensus_engine.calculate_consensus(
            reg_rating=reg_rating,
            strat_rating=new_rating,
            strat_confidence=new_score,
            reg_rationale=reg_rationale,
            strat_rationale=strat_rationale
        )

        final_rating = consensus_result.final_rating
        conviction_score = consensus_result.conviction_score

        # 4. Facility Assessment (Shared Logic)
        sorted_cap_structure = sorted(capital_structure, key=lambda x: x.get('priority', 99))
        primary_facility = sorted_cap_structure[0] if sorted_cap_structure else {'name': 'General Facility', 'amount': total_debt}

        collateral_bucket = "Weak"
        if ltv < 0.6: collateral_bucket = "Strong"
        elif ltv < 0.9: collateral_bucket = "Adequate"

        # 5. Construct Result with Dialogue
        result = SNCRatingModel(
            overall_borrower_rating=final_rating,
            rationale=consensus_result.narrative,
            primary_facility_assessment=PrimaryFacilityAssessment(
                facility_type=primary_facility.get('name', 'Unknown'),
                collateral_coverage=collateral_bucket,
                repayment_capacity="High" if final_rating == "Pass" else "Low"
            ),
            legacy_rating=reg_rating,
            model_consensus=(reg_rating == new_rating),
            conviction_score=conviction_score,
            risk_dialogue=consensus_result.risk_dialogue
        )

        logger.info(f"Consensus Reached: {final_rating} (Conviction: {conviction_score:.2f})")
        return result

from __future__ import annotations
from typing import Any, Dict, List
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import SNCRatingModel, Facility, CreditAnalysis, CovenantRiskAnalysis

class SNCRatingAgent(AgentBase):
    """
    Agent responsible for Shared National Credit (SNC) rating analysis.
    Implements Phase 3 of the 'Deep Dive' Pipeline.
    """
    async def execute(self, credit_data: Dict[str, Any]) -> CreditAnalysis:
        """
        Executes the SNC rating logic.

        Args:
            credit_data: Dictionary containing:
                - facilities: List of dicts with 'id', 'amount', 'ltv', 'interest_coverage'
                - primary_constraint: str
                - current_leverage: float
                - covenant_threshold: float
                - cds_rating: str
        """
        facilities_data = credit_data.get("facilities", [])
        facilities = []

        # If no facilities provided, create a dummy one for structure validity or handle gracefully
        if not facilities_data:
            # For robustness, we might log a warning but return a default structure
            # But strictly, we should probably return what we can.
            pass

        for f in facilities_data:
            ltv = f.get("ltv", 0.5)
            icr = f.get("interest_coverage", 4.0)

            rating = self._calculate_rating(ltv, icr)

            facilities.append(Facility(
                id=f.get("id", "Unknown Facility"),
                amount=str(f.get("amount", "0")),
                regulatory_rating=rating,
                collateral_coverage=f"{ltv:.1%}",
                covenant_headroom=f.get("covenant_headroom", "Unknown")
            ))

        overall_rating = self._calculate_overall_rating(facilities)

        snc_model = SNCRatingModel(
            overall_borrower_rating=overall_rating,
            facilities=facilities
        )

        covenant_analysis = CovenantRiskAnalysis(
            primary_constraint=credit_data.get("primary_constraint", "Net Leverage"),
            current_level=credit_data.get("current_leverage", 0.0),
            breach_threshold=credit_data.get("covenant_threshold", 4.0),
            risk_assessment=self._assess_covenant_risk(
                credit_data.get("current_leverage", 0.0),
                credit_data.get("covenant_threshold", 4.0)
            )
        )

        return CreditAnalysis(
            snc_rating_model=snc_model,
            cds_market_implied_rating=credit_data.get("cds_rating", "BB"),
            covenant_risk_analysis=covenant_analysis
        )

    def _calculate_rating(self, ltv: float, icr: float) -> str:
        """
        Calculates Regulatory Rating based on LTV and Interest Coverage Ratio.
        """
        if ltv < 0.60 and icr > 3.0:
            return "Pass"
        elif ltv < 0.80 and icr > 2.0:
            return "Special Mention"
        elif ltv < 1.0 and icr > 1.0:
            return "Substandard"
        else:
            return "Doubtful"

    def _calculate_overall_rating(self, facilities: List[Facility]) -> str:
        """
        Aggregates facility ratings into an overall borrower rating.
        Conservative approach: takes the worst rating of the facilities.
        """
        if not facilities:
            return "Pass" # Default if no facilities

        ratings_severity = {
            "Pass": 1,
            "Special Mention": 2,
            "Substandard": 3,
            "Doubtful": 4
        }

        worst_rating = "Pass"
        max_severity = 0

        for f in facilities:
            severity = ratings_severity.get(f.regulatory_rating, 0)
            if severity > max_severity:
                max_severity = severity
                worst_rating = f.regulatory_rating

        return worst_rating

    def _assess_covenant_risk(self, current: float, threshold: float) -> str:
        if threshold == 0:
            return "Unknown"
        headroom = (threshold - current) / threshold
        if headroom < 0:
            return "Breach"
        elif headroom < 0.1:
            return "Critical"
        elif headroom < 0.2:
            return "High"
        else:
            return "Low"

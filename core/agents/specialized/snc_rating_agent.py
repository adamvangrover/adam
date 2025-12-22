import logging
from typing import Dict, Any, List, Optional
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import SNCRatingModel, PrimaryFacilityAssessment

# Configure logging
logger = logging.getLogger(__name__)


class SNCRatingAgent(AgentBase):
    """
    Specialized Agent for performing Shared National Credit (SNC) simulations.

    Acts as a virtual 'Senior Credit Officer', applying regulatory frameworks
    (OCC/Fed/FDIC) to classify debt facilities based on:
    1. Primary Repayment Source (Cash Flow/EBITDA)
    2. Secondary Repayment Source (Collateral/Enterprise Value)

    Developer Note:
    ---------------
    This agent implements the "Interagency Guidance on Leveraged Lending" logic.
    It separates the borrower-level rating (Ability to Repay) from the facility-level
    rating (Loss Given Default), allowing for "notching up" based on collateral.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.persona = "SNC Examiner"

    def execute(self,
                financials: Dict[str, float],
                capital_structure: List[Dict[str, Any]],
                enterprise_value: float) -> SNCRatingModel:
        """
        Executes the SNC rating classification logic.

        Args:
            financials: Dict containing 'ebitda', 'interest_expense', 'total_debt', 'free_cash_flow'.
            capital_structure: List of facility dicts {'name': str, 'amount': float, 'priority': int, 'security': str}.
            enterprise_value: Current estimated EV for collateral coverage analysis.

        Returns:
            SNCRatingModel: Pydantic model containing the regulatory rating and facility details.
        """
        logger.info("Initiating SNC Regulatory Simulation...")

        # 1. Extract Key Metrics
        ebitda = float(financials.get('ebitda', 0.0))
        total_debt = float(financials.get('total_debt', 0.0))
        interest_expense = float(financials.get('interest_expense', 1.0))  # Avoid div/0

        # Validation checks
        if interest_expense <= 0:
            interest_expense = 1.0

        # 2. Calculate Borrower-Level Risk Ratios
        # Leverage: Total Debt / EBITDA
        leverage_ratio = total_debt / ebitda if ebitda > 0 else 99.0

        # Coverage: EBITDA / Interest Expense
        interest_coverage = ebitda / interest_expense

        logger.debug(f"SNC Metrics: Leverage={leverage_ratio:.2f}x, Coverage={interest_coverage:.2f}x")

        # 3. Determine Borrower Base Rating (The "Ability to Repay")
        # Logic mimics the Interagency Guidance on Leveraged Lending
        # Pass: Leverage < 4x, Coverage > 3x
        # Special Mention: Leverage 4-6x, Coverage 1.5-3x (Potential weakness)
        # Substandard: Leverage > 6x OR Coverage < 1.5x (Well-defined weakness, payment jeopardy)
        # Doubtful: Leverage > 8x (Collection highly improbable)

        if leverage_ratio < 4.0 and interest_coverage > 3.0:
            borrower_rating = "Pass"
        elif leverage_ratio < 6.0 and interest_coverage > 1.5:
            borrower_rating = "Special Mention"
        elif leverage_ratio < 8.0 or interest_coverage < 1.0:
            borrower_rating = "Substandard"
        else:
            borrower_rating = "Doubtful"

        # 4. Facility-Level Analysis (Focus on Primary Facility)

        # Sort capital structure by priority (1 = Senior Secured)
        sorted_cap_structure = sorted(capital_structure, key=lambda x: x.get('priority', 99))

        # Select Primary Facility (Largest or Senior)
        primary_facility = sorted_cap_structure[0] if sorted_cap_structure else {
            'name': 'General Facility', 'amount': total_debt}

        facility_name = primary_facility.get('name', 'Unknown Facility')
        cumulative_debt = float(primary_facility.get('amount', 0.0))  # Simplified for primary only logic

        # Calculate Collateral Coverage (EV / Cumulative Debt through this tranche)
        collateral_coverage_ratio = enterprise_value / cumulative_debt if cumulative_debt > 0 else 0.0

        # Determine Collateral Bucket
        collateral_bucket = "Weak"
        if collateral_coverage_ratio > 1.5:
            collateral_bucket = "Strong"
        elif collateral_coverage_ratio > 1.0:
            collateral_bucket = "Adequate"

        # 5. Construct Final Output
        result = SNCRatingModel(
            overall_borrower_rating=borrower_rating,
            rationale=f"Leverage: {leverage_ratio:.1f}x, Coverage: {interest_coverage:.1f}x",
            primary_facility_assessment=PrimaryFacilityAssessment(
                facility_type=facility_name,
                collateral_coverage=collateral_bucket,
                repayment_capacity="High" if borrower_rating == "Pass" else "Low"
            )
        )

        logger.info(f"SNC Simulation Complete. Borrower Rating: {borrower_rating}")
        return result

    def _estimate_covenant_headroom(self, current_leverage: float, rating: str) -> str:
        """
        Estimates the headroom (cushion) against maintenance covenants.

        Logic:
        - "Pass" borrowers typically have looser covenants (High headroom).
        - "Substandard" borrowers are likely tight or in breach.
        """
        # Base assumption: Market standard covenant is set 25-30% above closing leverage.
        # However, for stressed credits, the covenant might be fixed.

        if rating == "Pass":
            implied_covenant = current_leverage * 1.35  # Healthy cushion
        elif rating == "Special Mention":
            implied_covenant = current_leverage * 1.15  # Tightening
        else:
            implied_covenant = current_leverage * 0.95  # Likely in breach or wavering

        headroom_pct = (implied_covenant - current_leverage) / implied_covenant if implied_covenant > 0 else 0.0

        if headroom_pct < 0:
            return f"POTENTIAL BREACH ({headroom_pct:.1%})"
        else:
            return f"{headroom_pct:.1%} (Est. Cushion)"

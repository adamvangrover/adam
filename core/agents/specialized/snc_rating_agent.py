import logging
from typing import Dict, Any, List, Optional
from core.agents.agent_base import AgentBase
from core.schemas.v23_5_schema import SNCRatingModel, Facility

# Configure logging
logger = logging.getLogger(__name__)

class SNCRatingAgent(AgentBase):
    """
    Specialized Agent for performing Shared National Credit (SNC) simulations.

    Acts as a virtual 'Senior Credit Officer', applying regulatory frameworks
    (OCC/Fed/FDIC) to classify debt facilities based on:
    1. Primary Repayment Source (Cash Flow/EBITDA)
    2. Secondary Repayment Source (Collateral/Enterprise Value)
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
        ebitda = financials.get('ebitda', 0.0)
        total_debt = financials.get('total_debt', 0.0)
        interest_expense = financials.get('interest_expense', 1.0) # Avoid div/0

        # 2. Calculate Borrower-Level Risk Ratios
        leverage_ratio = total_debt / ebitda if ebitda > 0 else 99.0
        interest_coverage = ebitda / interest_expense

        logger.debug(f"SNC Metrics: Leverage={leverage_ratio:.2f}x, Coverage={interest_coverage:.2f}x")

        # 3. Determine Borrower Base Rating (The "Ability to Repay")
        # Logic mimics the Interagency Guidance on Leveraged Lending
        if leverage_ratio < 4.0 and interest_coverage > 3.0:
            borrower_rating = "Pass"
        elif leverage_ratio < 6.0 and interest_coverage > 1.5:
            borrower_rating = "Special Mention"
        elif leverage_ratio < 8.0 or interest_coverage < 1.0:
            borrower_rating = "Substandard"
        else:
            borrower_rating = "Doubtful"

        # 4. Facility-Level Analysis (waterfall for collateral coverage)
        rated_facilities = []
        cumulative_debt = 0.0

        # Sort capital structure by priority (1 = Senior Secured)
        sorted_cap_structure = sorted(capital_structure, key=lambda x: x.get('priority', 99))

        for debt in sorted_cap_structure:
            facility_name = debt.get('name', 'Unknown Facility')
            amount = debt.get('amount', 0.0)
            cumulative_debt += amount

            # Calculate Collateral Coverage (EV / Cumulative Debt through this tranche)
            # If EV covers the debt fully, it supports a better rating even if cash flow is weak.
            collateral_coverage_ratio = enterprise_value / cumulative_debt if cumulative_debt > 0 else 0.0

            facility_rating = borrower_rating

            # Notching Up logic: If collateral coverage is strong (>1.0x) explicitly for this tranche,
            # we can notch up a "Substandard" borrower to a "Pass" or "Special Mention" facility.
            if collateral_coverage_ratio >= 1.2 and borrower_rating in ["Substandard", "Doubtful"]:
                facility_rating = "Special Mention" # "Asset-based Pass" logic

            # Formatting for Schema
            collateral_str = f"{collateral_coverage_ratio:.2f}x EV Coverage"
            covenant_headroom = self._estimate_covenant_headroom(leverage_ratio)

            rated_facilities.append(
                Facility(
                    id=facility_name,
                    amount=f"${amount:,.2f}M",
                    regulatory_rating=facility_rating,
                    collateral_coverage=collateral_str,
                    covenant_headroom=covenant_headroom
                )
            )

        # 5. Construct Final Output
        result = SNCRatingModel(
            overall_borrower_rating=borrower_rating,
            facilities=rated_facilities
        )

        logger.info(f"SNC Simulation Complete. Borrower Rating: {borrower_rating}")
        return result

    def _estimate_covenant_headroom(self, current_leverage: float) -> str:
        # Simple heuristic: Assume standard Maintenance Covenant is 0.5x-1.0x above closing leverage
        # In a real scenario, CovenantAnalystAgent would parse the Credit Agreement.
        market_std_covenant = current_leverage + 1.0
        headroom = (market_std_covenant - current_leverage) / market_std_covenant
        return f"{headroom:.1%} (Est.)"

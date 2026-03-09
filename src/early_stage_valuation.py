# src/early_stage_valuation.py
from typing import Dict, Any, Tuple

class EarlyStageModel:
    """
    Framework for valuing investments or lending to early-stage high-growth pre-IPO companies.
    Incorporates implied valuation, liquidity runway, and interim covenant protection until EBITDA/FCF positive.
    """
    def __init__(
        self,
        current_cash: float,
        monthly_burn_rate: float,
        projected_ebitda_positive_months: int,
        target_ebitda: float,
        ebitda_multiple: float,
        discount_rate: float,
        total_debt: float = 0.0
    ):
        """
        Initializes the early-stage valuation model.

        Args:
            current_cash: Current cash reserves on the balance sheet.
            monthly_burn_rate: Average monthly cash burn (positive number indicating cash out).
            projected_ebitda_positive_months: Number of months until the company is projected to be EBITDA positive.
            target_ebitda: Projected annualized EBITDA at the time of turning positive.
            ebitda_multiple: Industry multiple applied to target EBITDA for valuation.
            discount_rate: Annual discount rate to discount future valuation to present.
            total_debt: Total outstanding debt (for LTV and EV to Equity Value conversion).
        """
        if current_cash < 0:
            raise ValueError("Current cash cannot be negative.")
        if monthly_burn_rate <= 0:
            raise ValueError("Monthly burn rate must be greater than zero.")
        if projected_ebitda_positive_months < 0:
            raise ValueError("Projected months to EBITDA positive cannot be negative.")
        if target_ebitda <= 0:
            raise ValueError("Target EBITDA must be positive for meaningful valuation.")
        if ebitda_multiple <= 0:
            raise ValueError("EBITDA multiple must be positive.")
        if discount_rate <= 0 or discount_rate >= 1:
            raise ValueError("Discount rate must be strictly between 0 and 1.")
        if total_debt < 0:
            raise ValueError("Total debt cannot be negative.")

        self.current_cash = float(current_cash)
        self.monthly_burn_rate = float(monthly_burn_rate)
        self.projected_ebitda_positive_months = int(projected_ebitda_positive_months)
        self.target_ebitda = float(target_ebitda)
        self.ebitda_multiple = float(ebitda_multiple)
        self.discount_rate = float(discount_rate)
        self.total_debt = float(total_debt)

    def calculate_liquidity_runway(self) -> float:
        """
        Calculates the liquidity runway in months.

        Returns:
            Number of months the company can survive on current cash reserves.
        """
        return self.current_cash / self.monthly_burn_rate

    def calculate_implied_valuation(self) -> Dict[str, float]:
        """
        Calculates the future implied enterprise value and discounts it back to present.

        Returns:
            Dictionary containing Future Enterprise Value and Present Enterprise Value.
        """
        future_ev = self.target_ebitda * self.ebitda_multiple
        # Discount back based on fractional years
        years_to_positive = self.projected_ebitda_positive_months / 12.0
        present_ev = future_ev / ((1 + self.discount_rate) ** years_to_positive)

        return {
            "Future Enterprise Value": round(future_ev, 2),
            "Present Enterprise Value": round(present_ev, 2)
        }

    def evaluate_ltv_framework(self, proposed_loan_amount: float = 0.0) -> Dict[str, float]:
        """
        Evaluates the Loan-to-Value (LTV) metric given a proposed additional loan amount
        against the implied present enterprise value.

        Args:
            proposed_loan_amount: Potential new lending amount.

        Returns:
            Dictionary containing Total Debt (including proposed) and LTV %.
        """
        if proposed_loan_amount < 0:
            raise ValueError("Proposed loan amount cannot be negative.")

        valuation_metrics = self.calculate_implied_valuation()
        present_ev = valuation_metrics["Present Enterprise Value"]
        
        total_pro_forma_debt = self.total_debt + proposed_loan_amount

        # Edge case: If present EV is virtually zero or negative (though constrained above)
        if present_ev <= 0:
            ltv = float('inf')
        else:
            ltv = total_pro_forma_debt / present_ev

        return {
            "Pro Forma Debt": round(total_pro_forma_debt, 2),
            "LTV (%)": round(ltv * 100, 2)
        }

    def check_interim_covenant_protection(self, minimum_liquidity_covenant: float) -> Tuple[bool, str]:
        """
        Assesses interim covenant protection by evaluating if the current cash
        satisfies minimum liquidity requirements to survive until projected EBITDA positive.
        
        Args:
            minimum_liquidity_covenant: The absolute minimum cash balance required at any time.

        Returns:
            Tuple of (covenant_pass_status (bool), detailed message (str)).
        """
        if minimum_liquidity_covenant < 0:
            raise ValueError("Minimum liquidity covenant cannot be negative.")

        runway_months = self.calculate_liquidity_runway()
        
        if self.current_cash < minimum_liquidity_covenant:
            return False, f"BREACH: Current cash ({self.current_cash}) is below minimum liquidity covenant ({minimum_liquidity_covenant})."

        # Check if runway is sufficient to reach EBITDA positive
        if runway_months < self.projected_ebitda_positive_months:
            shortfall_months = self.projected_ebitda_positive_months - runway_months
            required_additional_cash = shortfall_months * self.monthly_burn_rate
            return False, f"WARNING/BREACH: Liquidity runway ({round(runway_months, 1)} months) is insufficient to reach projected EBITDA positive in {self.projected_ebitda_positive_months} months. Additional capital required: ~{round(required_additional_cash, 2)}."

        return True, "PASS: Adequate liquidity runway and minimum cash covenant satisfied until projected EBITDA positive."

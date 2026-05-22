# src/early_stage_valuation.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple

class EarlyStageModel(BaseModel):
    """
    [AI-CONTEXT: Framework for valuing early-stage high-growth pre-IPO companies.
    Validates inputs automatically via Pydantic v2. Provides liquidity runway,
    implied present enterprise value, LTV metrics, and covenant protection checks.]
    """
    current_cash: float = Field(..., ge=0.0, description="Current cash reserves on the balance sheet.")
    monthly_burn_rate: float = Field(..., gt=0.0, description="Average monthly cash burn (positive number).")
    projected_ebitda_positive_months: int = Field(..., ge=0, description="Months until company is EBITDA positive.")
    target_ebitda: float = Field(..., gt=0.0, description="Projected annualized EBITDA at the time of turning positive.")
    ebitda_multiple: float = Field(..., gt=0.0, description="Industry multiple applied to target EBITDA for valuation.")
    discount_rate: float = Field(..., gt=0.0, lt=1.0, description="Annual discount rate.")
    total_debt: float = Field(0.0, ge=0.0, description="Total outstanding debt.")

    @property
    def liquidity_runway(self) -> float:
        """Calculates the liquidity runway in months."""
        return self.current_cash / self.monthly_burn_rate

    def calculate_liquidity_runway(self) -> float:
        """
        Legacy compatibility method for calculating liquidity runway.

        Returns:
            Number of months the company can survive on current cash reserves.
        """
        return self.liquidity_runway

    def calculate_implied_valuation(self) -> dict[str, float]:
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

    def evaluate_ltv_framework(self, proposed_loan_amount: float = 0.0) -> dict[str, float]:
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

        # Edge case: If present EV is virtually zero or negative
        ltv = float('inf') if present_ev <= 0 else total_pro_forma_debt / present_ev

        return {
            "Pro Forma Debt": round(total_pro_forma_debt, 2),
            "LTV (%)": round(ltv * 100, 2)
        }

    def check_interim_covenant_protection(self, minimum_liquidity_covenant: float) -> tuple[bool, str]:
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

        runway_months = self.liquidity_runway
        
        if self.current_cash < minimum_liquidity_covenant:
            return False, f"BREACH: Current cash ({self.current_cash}) is below minimum liquidity covenant ({minimum_liquidity_covenant})."

        # Check if runway is sufficient to reach EBITDA positive
        if runway_months < self.projected_ebitda_positive_months:
            shortfall_months = self.projected_ebitda_positive_months - runway_months
            required_additional_cash = shortfall_months * self.monthly_burn_rate
            return False, f"WARNING/BREACH: Liquidity runway ({round(runway_months, 1)} months) is insufficient to reach projected EBITDA positive in {self.projected_ebitda_positive_months} months. Additional capital required: ~{round(required_additional_cash, 2)}."

        return True, "PASS: Adequate liquidity runway and minimum cash covenant satisfied until projected EBITDA positive."

    def generate_risk_summary(self, proposed_loan_amount: float = 0.0, minimum_liquidity_covenant: float = 0.0) -> str:
        """
        [AI-INNOVATOR] Generates a lightweight, AI-readable risk profile summarizing runway risk,
        implied valuation, and debt coverage.

        Returns:
            String payload containing actionable risk metrics for autonomous evaluation.
        """
        val = self.calculate_implied_valuation()
        ltv_metrics = self.evaluate_ltv_framework(proposed_loan_amount)
        cov_pass, cov_msg = self.check_interim_covenant_protection(minimum_liquidity_covenant)

        status = "SAFE" if cov_pass and ltv_metrics["LTV (%)"] < 80.0 else "AT_RISK"
        return (f"[RISK:{status}] Runway: {round(self.liquidity_runway, 1)}m | "
                f"LTV: {ltv_metrics['LTV (%)']}% | "
                f"Present EV: {val['Present Enterprise Value']} | "
                f"Covenant Status: {cov_msg}")

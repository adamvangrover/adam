import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RatioCalculator:
    """
    Deterministic calculation engine for financial ratios.
    System 2 Component: Logic > Hallucination.
    """

    def calculate_all(self, financials: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates a standard suite of credit ratios.

        Args:
            financials: Dict containing:
                - ebitda
                - total_debt
                - interest_expense
                - total_assets
                - total_liabilities
                - total_equity
                - current_assets
                - current_liabilities
                - net_income
        """
        results = {}
        results['interest_coverage'] = self.calculate_coverage(
            financials.get('ebitda'), financials.get('interest_expense')
        )
        results['leverage'] = self.calculate_leverage(
            financials.get('total_debt'), financials.get('ebitda')
        )
        results['debt_to_equity'] = self.calculate_dte(
            financials.get('total_debt'), financials.get('total_equity')
        )
        results['current_ratio'] = self.calculate_ratio(
            financials.get('current_assets'), financials.get('current_liabilities')
        )
        results['roa'] = self.calculate_ratio(
            financials.get('net_income'), financials.get('total_assets')
        )

        # Remove None values
        return {k: v for k, v in results.items() if v is not None}

    def calculate_ratio(self, numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        """Safe division."""
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return 0.0 # Or float('inf') depending on preference, staying safe with 0
        return round(numerator / denominator, 4)

    def calculate_coverage(self, ebitda: Optional[float], interest: Optional[float]) -> Optional[float]:
        """Calculates Interest Coverage Ratio (EBITDA / Interest)."""
        return self.calculate_ratio(ebitda, interest)

    def calculate_leverage(self, debt: Optional[float], ebitda: Optional[float]) -> Optional[float]:
        """Calculates Leverage Ratio (Total Debt / EBITDA)."""
        return self.calculate_ratio(debt, ebitda)

    def calculate_dte(self, debt: Optional[float], equity: Optional[float]) -> Optional[float]:
        """Calculates Debt-to-Equity Ratio."""
        return self.calculate_ratio(debt, equity)

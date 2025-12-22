from typing import Dict, Any
from ..state import VerticalRiskGraphState


class QuantAgent:
    """
    Specialized in extracting tabular data, running calculations, and analyzing Excel models.
    """

    def __init__(self, model):
        self.model = model

    def analyze_financials(self, state: VerticalRiskGraphState) -> Dict[str, Any]:
        """
        Extracts financial data and calculates ratios.
        """
        print("--- Quant Agent: Analyzing Financials ---")

        # Extract data from state or use defaults (from XBRLHandler mock if empty)
        bs = state.get("balance_sheet") or {}
        pnl = state.get("income_statement") or {}

        # Robust retrieval with defaults
        cash = bs.get("cash_equivalents", 0.0)
        debt = bs.get("total_debt", 0.0)

        revenue = pnl.get("revenue", 0.0)
        op_inc = pnl.get("operating_income", 0.0)
        da = pnl.get("depreciation_amortization", 0.0)
        interest = pnl.get("interest_expense", 1.0)  # Avoid div/0
        if interest == 0:
            interest = 1.0

        # Calculate EBITDA
        ebitda = pnl.get("consolidated_ebitda")
        if not ebitda:
            ebitda = op_inc + da

        # Ratios
        leverage = debt / ebitda if ebitda else 0.0
        net_leverage = (debt - cash) / ebitda if ebitda else 0.0
        coverage = ebitda / interest

        margin = ebitda / revenue if revenue else 0.0

        # Stress Test (Scenario: -20% EBITDA)
        stressed_ebitda = ebitda * 0.8
        stressed_leverage = debt / stressed_ebitda if stressed_ebitda else 0.0

        analysis_text = (
            f"Financial Analysis:\n"
            f"- EBITDA: ${ebitda:,.2f} (Margin: {margin:.1%})\n"
            f"- Leverage (Gross): {leverage:.2f}x\n"
            f"- Leverage (Net): {net_leverage:.2f}x\n"
            f"- Interest Coverage: {coverage:.2f}x\n"
            f"\nStress Test (-20% EBITDA):\n"
            f"- Stressed Leverage: {stressed_leverage:.2f}x"
        )

        return {
            "quant_analysis": analysis_text,
            "balance_sheet": bs,
            "income_statement": pnl,  # Pass through
            "messages": ["Quant: Financial analysis and stress testing complete."]
        }

from typing import Dict, Any, List

class ZombieFactory:
    """
    The 'Red Team' layer.
    Generates adversarial scenarios ("Zombie Companies") to stress-test the agents.
    """

    @staticmethod
    def generate_zombie_state() -> Dict[str, Any]:
        """
        Creates a 'Zombie Company' state designed to fail risk assessments.
        - Leverage: ~13.3x (400M Debt / 30M EBITDA)
        - Coverage: ~0.66x (30M EBITDA / 45M Interest)
        - Includes metadata to trigger Symbolic checks.
        """
        return {
            "ticker": "ZOMB",
            "data_room_path": "/mock/zombie",
            "balance_sheet": {
                "cash_equivalents": 5_000_000,
                "total_assets": 450_000_000,
                "total_debt": 400_000_000,  # Massive Debt Load
                "equity": 45_000_000,
                "currency": "USD",
                "fiscal_year": 2024
            },
            "income_statement": {
                "revenue": 100_000_000,
                "operating_income": 20_000_000,
                "net_income": -20_000_000,
                "depreciation_amortization": 10_000_000,
                "interest_expense": 45_000_000, # Crushing Interest
                "consolidated_ebitda": 30_000_000
            },
            "covenants": [
                 {
                     "name": "Net Leverage Ratio",
                     "threshold": 4.5,
                     "operator": "<=",
                     "definition_text": "Total Net Debt / Consolidated EBITDA",
                     "add_backs": []
                 }
            ],
            # Initialize empty fields
            "quant_analysis": None,
            "legal_analysis": None,
            "market_research": None,
            "draft_memo": None,
            "messages": [],
            "critique_count": 0,
            "status": "red_team_test",
            "audit_logs": [],
            "verification_flags": []
        }

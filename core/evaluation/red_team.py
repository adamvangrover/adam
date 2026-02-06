from typing import Dict, Any, List
import copy

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

    @staticmethod
    def generate_sensitivity_scenarios(base_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates a suite of stress-test scenarios by tweaking key variables.
        Variations:
        1. EBITDA -20%
        2. Interest Expense +200bps (approx +20% cost)
        3. Revenue -15% (Recession)
        """
        scenarios = []

        # Scenario 1: EBITDA Shock
        s1 = copy.deepcopy(base_state)
        orig_ebitda = s1["income_statement"].get("consolidated_ebitda", 0)
        s1["income_statement"]["consolidated_ebitda"] = orig_ebitda * 0.8
        s1["scenario_id"] = "EBITDA_SHOCK_20PCT"
        scenarios.append(s1)

        # Scenario 2: Rate Hike
        s2 = copy.deepcopy(base_state)
        orig_interest = s2["income_statement"].get("interest_expense", 0)
        s2["income_statement"]["interest_expense"] = orig_interest * 1.25
        s2["scenario_id"] = "RATE_HIKE_250BPS"
        scenarios.append(s2)

        # Scenario 3: Revenue Collapse
        s3 = copy.deepcopy(base_state)
        orig_rev = s3["income_statement"].get("revenue", 0)
        s3["income_statement"]["revenue"] = orig_rev * 0.85
        # Assume 40% flow through to EBITDA
        delta_rev = orig_rev * 0.15
        s3["income_statement"]["consolidated_ebitda"] = orig_ebitda - (delta_rev * 0.4)
        s3["scenario_id"] = "REVENUE_COLLAPSE_15PCT"
        scenarios.append(s3)

        return scenarios

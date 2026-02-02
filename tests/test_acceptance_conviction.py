import unittest
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.sector_impact_engine import SectorImpactEngine

class TestAcceptanceConviction(unittest.TestCase):
    """
    System Acceptance Level Testing:
    Validates that the 'SectorImpactEngine' output aligns with the
    qualitative 'High Conviction' ratings from the Strategic Outlook report.
    """

    def setUp(self):
        self.engine = SectorImpactEngine()
        # Override context: Remove themes that punish Tech to see pure Scenario impact.
        self.engine.context = {
            "snapshot_date": "2026-01-30",
            "macro_regime": "Bifurcated Normalization",
            "themes": [] # No active adverse themes
        }
        self.scenario_id = "SCENARIO_BIFURCATED_NORMALIZATION"

        # Define the portfolio based on the report
        self.portfolio = [
            # High Conviction (Expect Low Risk Score)
            {"name": "Amazon", "sector": "Technology", "leverage": 3.0, "rating": "A"},
            {"name": "Oracle", "sector": "Technology", "leverage": 4.5, "rating": "BBB"},
            {"name": "Nvidia", "sector": "Technology", "leverage": 1.0, "rating": "A"},

            # Medium Conviction
            {"name": "Adobe", "sector": "Technology", "leverage": 2.0, "rating": "A-"},

            # Defensive
            {"name": "AngloGold", "sector": "Materials", "leverage": 3.0, "rating": "BB"}
        ]

    def test_conviction_alignment(self):
        """
        Critique: Does the simulation align with 'High Conviction' ratings?
        In this engine: Lower Score = Lower Risk = Better.
        """
        results_obj = self.engine.analyze_portfolio(self.portfolio, scenario_id=self.scenario_id)
        results = {r['asset']: r for r in results_obj['results']}

        # 1. Oracle & Nvidia (High Conviction) should be Low Risk
        # Tech Geo Risk is hardcoded to 75.0, so scores hover around 30.0 best case.
        # Nvidia (Lev 1): Macro(15) + Credit(0) + Geo(75) = 90 / 3 = 30.0.

        orcl_score = results['Oracle']['consensus_score']
        nvda_score = results['Nvidia']['consensus_score']

        # We assert they are in the "Investable" range (< 50).
        self.assertLessEqual(orcl_score, 50.0, f"Oracle Risk Score {orcl_score} is too high")
        self.assertLessEqual(nvda_score, 45.0, f"Nvidia Risk Score {nvda_score} is too high")

        # 2. AngloGold (Hedge)
        # Materials (+0.4 shock).
        au_score = results['AngloGold']['consensus_score']
        self.assertLessEqual(au_score, 50.0, "AngloGold should be a safe hedge")

    def test_bifurcation_within_tech(self):
        """
        Verify 'Bifurcated Normalization' by comparing Quality Tech vs Distressed Tech.
        The scenario should favor the Quality asset significantly.
        """
        test_assets = [
            {"name": "Nvidia", "sector": "Technology", "leverage": 1.0, "rating": "AAA"},
            {"name": "LegacyDebtCo", "sector": "Technology", "leverage": 6.0, "rating": "CCC"}
        ]

        res_obj = self.engine.analyze_portfolio(test_assets, scenario_id=self.scenario_id)
        res_map = {r['asset']: r['consensus_score'] for r in res_obj['results']}

        # Nvidia should have significantly lower risk than LegacyDebtCo
        print(f"Nvidia Score: {res_map['Nvidia']}")
        print(f"Legacy Score: {res_map['LegacyDebtCo']}")

        self.assertLess(res_map['Nvidia'], res_map['LegacyDebtCo'],
                        "Quality Tech should outperform Legacy Distressed Tech")

        # Verify the spread is substantial (> 15 points)
        self.assertGreater(res_map['LegacyDebtCo'] - res_map['Nvidia'], 15.0,
                           "Bifurcation spread is too narrow")

if __name__ == '__main__':
    unittest.main()

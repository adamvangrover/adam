import unittest
import sys
import os
import pandas as pd

# Add root to sys.path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core_valuation import ValuationEngine
from src.credit_risk import CreditSponsorModel
from src.config import DEFAULT_ASSUMPTIONS

class TestFinancialPlatform(unittest.TestCase):
    def test_valuation_engine(self):
        val = ValuationEngine(ebitda_base=100, capex_percent=0.05, nwc_percent=0.02, debt_cost=0.05, equity_percent=0.4)
        df_proj, ev, wacc = val.run_dcf([0.05]*5)

        self.assertIsInstance(df_proj, pd.DataFrame)
        self.assertEqual(len(df_proj), 5)
        self.assertGreater(ev, 0)
        self.assertGreater(wacc, 0)

    def test_credit_risk_model(self):
        cred = CreditSponsorModel(enterprise_value=1000, total_debt=500, ebitda=100, interest_expense=25)
        metrics = cred.calculate_metrics()
        rating = cred.determine_regulatory_rating(metrics)

        self.assertIn("Leverage (x)", metrics)
        # Based on 500/100 = 5.0x Lev, FCCR 90/25 = 3.6.
        # Lev 5.0 is < 6.0, FCCR 3.6 > 1.1 -> RATING_MAP[5.0] which is "Pass 5 (B+)"
        self.assertEqual(rating, "Pass 5 (B+)")

if __name__ == '__main__':
    unittest.main()

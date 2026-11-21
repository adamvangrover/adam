import unittest
import sys
import os

# Bypass core.simulations.__init__ to avoid dependencies
sys.path.append(os.path.join(os.getcwd(), 'core', 'simulations'))
import comprehensive_credit_simulation as ccs

class TestComprehensiveCredit(unittest.TestCase):

    def test_abl_borrowing_base(self):
        """Test Asset Based Lending calculations."""
        sim = ccs.ComprehensiveCreditSimulation()
        assets = [
            ccs.CollateralAsset("AR", 100.0, ineligible_amount=10.0, advance_rate=0.85),
            ccs.CollateralAsset("Inventory", 50.0, ineligible_amount=5.0, advance_rate=0.60)
        ]

        bb, details = sim.calculate_borrowing_base(assets)

        # AR: (100 - 10) * 0.85 = 90 * 0.85 = 76.5
        # Inv: (50 - 5) * 0.60 = 45 * 0.60 = 27.0
        # Total: 103.5

        self.assertAlmostEqual(bb, 103.5)
        self.assertEqual(details["AR"]["net"], 90.0)

    def test_cash_flow_metrics(self):
        """Test EBITDA, Leverage, and DSCR."""
        sim = ccs.ComprehensiveCreditSimulation()
        debt = [
            ccs.LoanTranche("Term", 200.0, 0.05, 1, "Term"),
            ccs.LoanTranche("Sub", 100.0, 0.10, 2, "Mezzanine")
        ]

        # Interest: (200*0.05) + (100*0.10) = 10 + 10 = 20
        # EBITDA: 50. Capex: 5.

        metrics = sim.calculate_cash_flow_metrics(ebitda=50.0, capex=5.0, tax_rate=0.25, debt_stack=debt)

        self.assertEqual(metrics["total_debt"], 300.0)
        self.assertEqual(metrics["leverage"], 6.0) # 300/50
        self.assertEqual(metrics["dscr"], 2.25) # (50-5)/20 = 45/20 = 2.25

    def test_cva_calculation(self):
        """Test Counterparty Risk."""
        sim = ccs.ComprehensiveCreditSimulation()
        pos = ccs.DerivativePosition("Swap", 100.0, mtm_value=10.0, counterparty_rating="BBB")

        # PD for BBB (default 0.02), LGD 0.6
        # Exposure = 10.0
        # CVA = 10 * 0.02 * 0.6 = 0.12

        cva = sim.calculate_counterparty_risk([pos], pd_lookup={"BBB": 0.02})
        self.assertAlmostEqual(cva["cva_charge"], 0.12)

    def test_avg_restructuring(self):
        """Test simulated AVG restructuring search."""
        sim = ccs.ComprehensiveCreditSimulation()
        debt = [ccs.LoanTranche("Senior", 100.0, 0.05, 1, "Term")]

        res = sim.run_avg_restructuring_search(enterprise_value=200.0, debt_stack=debt)

        # Consensual EV (95% of 200) = 190
        # Litigation EV (80% of 200) = 160
        # Value Added = 30

        self.assertAlmostEqual(res["optimized_ev"], 190.0)
        self.assertAlmostEqual(res["value_added"], 30.0)

if __name__ == '__main__':
    unittest.main()

import unittest
import sys
import os

# Bypass core.simulations.__init__ to avoid dependencies
sys.path.append(os.path.join(os.getcwd(), 'core', 'simulations'))
import distressed_credit_pricing_simulation as sim_mod

class TestDistressedCreditPricing(unittest.TestCase):

    def test_waterfall_full_recovery(self):
        """Test that everyone gets paid if EV > Debt."""
        sim = sim_mod.DistressedCreditPricingSimulation()
        cs = sim_mod.CapitalStructure()
        cs.add_tranche(sim_mod.Tranche("Senior", 100, 0.05, 1))
        cs.add_tranche(sim_mod.Tranche("Junior", 50, 0.10, 2))

        # EV = 200 (Debt = 150)
        res = sim.simulate_restructuring(enterprise_value=200, cap_structure=cs)

        self.assertEqual(res["waterfall"][0]["recovery_pct"], 1.0)
        self.assertEqual(res["waterfall"][1]["recovery_pct"], 1.0)
        self.assertEqual(res["equity_recovery"], 50.0)

    def test_expected_loss_calculation(self):
        """Test EL calculation with Collateral."""
        sim = sim_mod.DistressedCreditPricingSimulation()
        cs = sim_mod.CapitalStructure()

        # Collateral: 50M PPE (Liquidated)
        cs.add_collateral(sim_mod.CollateralPool("PPE", 62.5, 0.8)) # 62.5 * 0.8 = 50.0

        # Senior Secured by PPE (100M Debt) -> 50M covered, 50M unsecured
        t1 = sim_mod.Tranche("Senior", 100, 0.05, 1, security_type="Secured", secured_by="PPE")
        cs.add_tranche(t1)

        # PD = 0.1
        res = sim.calculate_expected_loss(cs, pd=0.1, accounting_standard="IFRS9")

        # Tranche 1 LGD:
        # Covered = 50. Uncovered = 50.
        # LGD % = 1 - (50/100) = 0.5
        # EL = 0.1 * 0.5 * 100 = 5.0

        t1_res = res[0]
        self.assertAlmostEqual(t1_res["lgd_pct"], 0.5)
        self.assertAlmostEqual(t1_res["el"], 5.0)

    def test_gaap_legacy_accounting(self):
        """Test GAAP Legacy logic (Incurred Loss)."""
        sim = sim_mod.DistressedCreditPricingSimulation()
        cs = sim_mod.CapitalStructure()
        cs.add_tranche(sim_mod.Tranche("Senior", 100, 0.05, 1))

        # PD = 0.1 (Unlikely) -> EL should be 0 under legacy incurred model assumption in code
        res = sim.calculate_expected_loss(cs, pd=0.1, accounting_standard="GAAP_Legacy")
        self.assertEqual(res[0]["el"], 0.0)

        # PD = 0.6 (Probable) -> EL booked
        res2 = sim.calculate_expected_loss(cs, pd=0.6, accounting_standard="GAAP_Legacy")
        self.assertGreater(res2[0]["el"], 0.0)

if __name__ == '__main__':
    unittest.main()

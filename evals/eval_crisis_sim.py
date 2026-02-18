import sys
import os
import unittest
import math

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.market_mayhem_crisis_sim import CrisisSimulator

def calculate_std_dev(data):
    n = len(data)
    if n <= 1: return 0.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return math.sqrt(variance)

class TestCrisisSimulator(unittest.TestCase):
    def test_determinism(self):
        """Test that the simulator produces deterministic results for the same seed."""
        sim1 = CrisisSimulator(seed="test_seed_123", risk_profile="BALANCED")
        res1 = sim1.run_simulation("2008_LEHMAN")

        sim2 = CrisisSimulator(seed="test_seed_123", risk_profile="BALANCED")
        res2 = sim2.run_simulation("2008_LEHMAN")

        self.assertEqual(res1["market"], res2["market"])
        self.assertEqual(res1["portfolio"], res2["portfolio"])
        self.assertEqual(res1["labels"], res2["labels"])

    def test_structure(self):
        """Test that the output structure is correct and lengths match."""
        sim = CrisisSimulator(seed="structure_test", risk_profile="BALANCED")
        res = sim.run_simulation("2020_COVID")

        self.assertIsNotNone(res)
        self.assertIn("scenario", res)
        self.assertIn("labels", res)
        self.assertIn("market", res)
        self.assertIn("portfolio", res)

        num_points = len(res["labels"])
        self.assertEqual(len(res["market"]), num_points)
        self.assertEqual(len(res["portfolio"]), num_points)

    def test_risk_profile_impact(self):
        """Test that AGGRESSIVE profile has higher volatility (std dev) than CONSERVATIVE."""
        scenario = "2022_INFLATION"

        sim_agg = CrisisSimulator(seed="risk_test", risk_profile="AGGRESSIVE")
        res_agg = sim_agg.run_simulation(scenario)

        sim_cons = CrisisSimulator(seed="risk_test", risk_profile="CONSERVATIVE")
        res_cons = sim_cons.run_simulation(scenario)

        # Calculate volatility of portfolio returns
        # Returns = (P_t - P_t-1) / P_t-1

        def get_returns(series):
            return [(series[i] - series[i-1])/series[i-1] for i in range(1, len(series))]

        returns_agg = get_returns(res_agg["portfolio"])
        returns_cons = get_returns(res_cons["portfolio"])

        vol_agg = calculate_std_dev(returns_agg)
        vol_cons = calculate_std_dev(returns_cons)

        print(f"Aggressive Volatility: {vol_agg:.5f}")
        print(f"Conservative Volatility: {vol_cons:.5f}")

        self.assertGreater(vol_agg, vol_cons, "Aggressive profile should be more volatile than Conservative")

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from core.risk_engine.engine import RiskEngine

class TestRiskEngineV2(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine()

        # Test Portfolio: 2 Assets
        # Asset A: $1M, 20% Vol
        # Asset B: $1M, 30% Vol
        self.portfolio = [
            {"id": "Asset_A", "market_value": 1_000_000, "volatility": 0.20},
            {"id": "Asset_B", "market_value": 1_000_000, "volatility": 0.30}
        ]

        # Correlation Matrix (0.5 correlation)
        self.correlation_matrix = [
            [1.0, 0.5],
            [0.5, 1.0]
        ]

    def test_component_var_summation(self):
        """
        Verify Euler's Theorem: Sum of Component VaR should equal Total VaR.
        """
        if not self.engine.numpy_enabled:
            print("Skipping Component VaR test (Numpy missing)")
            return

        # Calculate Total VaR
        total_risk = self.engine.calculate_parametric_var(
            portfolio=self.portfolio,
            correlation_matrix=self.correlation_matrix,
            confidence_level=0.95
        )
        total_var_daily = total_risk["VaR_Daily"]

        # Calculate Component VaR
        comp_risk = self.engine.calculate_component_var(
            portfolio=self.portfolio,
            correlation_matrix=self.correlation_matrix,
            confidence_level=0.95
        )

        contributions = comp_risk["contributions"]
        sum_components = sum(contributions.values())

        print(f"\n[Test Component VaR] Total: {total_var_daily}, Sum Components: {sum_components}")

        # Allow small float error
        self.assertAlmostEqual(total_var_daily, sum_components, delta=1.0)

    def test_monte_carlo_convergence(self):
        """
        Verify Monte Carlo converges to Parametric VaR for Normal Distribution.
        """
        if not self.engine.numpy_enabled:
            print("Skipping Monte Carlo test (Numpy missing)")
            return

        # Parametric
        param_risk = self.engine.calculate_parametric_var(
            portfolio=self.portfolio,
            correlation_matrix=self.correlation_matrix,
            confidence_level=0.95
        )
        param_var = param_risk["VaR_Daily"]

        # Monte Carlo (Use enough sims for convergence)
        mc_risk = self.engine.simulate_monte_carlo_var(
            portfolio=self.portfolio,
            simulations=50000,
            correlation_matrix=self.correlation_matrix,
            confidence_level=0.95
        )
        mc_var = mc_risk["VaR_MonteCarlo"]

        print(f"\n[Test Monte Carlo] Parametric: {param_var}, MC: {mc_var}")

        # 5% tolerance is acceptable for Monte Carlo vs Parametric
        diff_pct = abs(mc_var - param_var) / param_var
        self.assertLess(diff_pct, 0.05)

    def test_stress_test(self):
        """
        Verify deterministic stress testing.
        """
        shocks = {
            "Asset_A": -0.10, # -10% -> Loss of 100k
            "Asset_B": 0.05   # +5%  -> Gain of 50k
        }

        result = self.engine.execute_stress_test(self.portfolio, shocks)

        expected_pnl = (1_000_000 * -0.10) + (1_000_000 * 0.05) # -50,000
        calc_pnl = result["Scenario_PnL"]

        print(f"\n[Test Stress] Expected: {expected_pnl}, Calculated: {calc_pnl}")
        self.assertEqual(calc_pnl, expected_pnl)

    def test_monte_carlo_fallback(self):
        """
        Verify that Monte Carlo handles non-positive definite matrices
        via Eigendecomposition fallback.
        """
        if not self.engine.numpy_enabled:
            print("Skipping Monte Carlo Fallback test (Numpy missing)")
            return

        # 3 Assets
        portfolio = [
            {"id": "A", "market_value": 100, "volatility": 0.2},
            {"id": "B", "market_value": 100, "volatility": 0.2},
            {"id": "C", "market_value": 100, "volatility": 0.2}
        ]

        # Inconsistent Correlation Matrix (Non-PD)
        # A-B: 0.9, B-C: 0.9, A-C: -0.9 (Impossible triangle)
        bad_matrix = [
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0]
        ]

        # Ensure it doesn't raise LinAlgError
        try:
            result = self.engine.simulate_monte_carlo_var(
                portfolio=portfolio,
                simulations=1000,
                correlation_matrix=bad_matrix
            )
            print(f"\n[Test MC Fallback] Result: {result.get('VaR_MonteCarlo')}")
            self.assertIn("VaR_MonteCarlo", result)
            self.assertGreater(result["VaR_MonteCarlo"], 0)
        except Exception as e:
            self.fail(f"Monte Carlo raised exception on non-PD matrix: {e}")

if __name__ == '__main__':
    unittest.main()

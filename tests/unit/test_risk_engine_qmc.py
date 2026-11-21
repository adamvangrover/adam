import unittest
import numpy as np
from core.risk_engine.quantum_monte_carlo import QuantumMonteCarloEngine, SimulatorBackend

class TestRiskEngineQMC(unittest.TestCase):
    def test_simulation_runs(self):
        """Test that the simulation runs without error and returns expected keys."""
        engine = QuantumMonteCarloEngine(n_simulations=100)
        result = engine.simulate_portfolio_risk(
            initial_value=1000,
            volatility=0.2,
            time_horizon=1.0
        )
        self.assertIn("VaR_95", result)
        self.assertIn("VaR_99", result)
        self.assertIn("Expected_Shortfall", result)
        self.assertIn("Quantum_Speedup_Factor", result)

    def test_vectorization_correctness(self):
        """Test that the vectorized backend returns correct shape."""
        backend = SimulatorBackend()
        n_simulations = 50
        results = backend.run_circuit_batch(n_simulations, 5, 10)
        self.assertEqual(results.shape, (n_simulations,))
        self.assertTrue(isinstance(results, np.ndarray))

    def test_statistical_properties(self):
        """Test that the statistical properties are roughly reasonable."""
        # Large N for statistical significance
        engine = QuantumMonteCarloEngine(n_simulations=10000)
        # Zero volatility should result in zero change
        result = engine.simulate_portfolio_risk(1000, 0.0, 1.0)
        self.assertAlmostEqual(result["VaR_95"], 0.0)

if __name__ == '__main__':
    unittest.main()

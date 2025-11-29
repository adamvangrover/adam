import unittest
import numpy as np
from core.xai.iqnn_cs import IQNNCS
from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
from core.v22_quantum_pipeline.qmc_engine import QuantumMonteCarloEngine

class TestQuantumCapabilities(unittest.TestCase):

    def test_iqnn_cs_functionality(self):
        """Test IQNN-CS interpretability metrics."""
        iqnn = IQNNCS(input_dim=5, num_classes=2)

        # Consistent Class 0
        for _ in range(5):
            iqnn.record_prediction(np.zeros(5), 0, np.array([1, 0, 0, 0, 0]))

        # Inconsistent Class 1
        iqnn.record_prediction(np.zeros(5), 1, np.array([0, 1, 0, 0, 0]))
        iqnn.record_prediction(np.zeros(5), 1, np.array([0, 0, 1, 0, 0]))

        scores = iqnn.calculate_icaa()
        self.assertAlmostEqual(scores[0], 1.0, places=4)
        self.assertLess(scores[1], 0.5)

    def test_generative_risk_engine(self):
        """Test Generative Risk Engine scenario generation."""
        engine = GenerativeRiskEngine()
        scenarios = engine.generate_scenarios(n_samples=10, regime="stress")
        self.assertEqual(len(scenarios), 10)
        self.assertTrue(any(s.scenario_id.startswith("stress") for s in scenarios))

        # Test Reverse Stress Test
        breaches = engine.reverse_stress_test(target_loss_threshold=1.0, current_portfolio_value=100.0)
        self.assertIsInstance(breaches, list)

    def test_qmc_engine(self):
        """Test Quantum Monte Carlo Engine (Mock)."""
        qmc = QuantumMonteCarloEngine()
        result = qmc.simulate_merton_model(
            asset_value=100, debt_face_value=90, volatility=0.2,
            risk_free_rate=0.05, time_horizon=1.0, n_simulations=100
        )
        self.assertIn("probability_of_default", result)
        self.assertIn("quantum_error_bound_theoretical", result)
        self.assertTrue(0 <= result["probability_of_default"] <= 1)

if __name__ == '__main__':
    unittest.main()

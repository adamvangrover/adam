import unittest
import numpy as np
from core.simulations.avg_search import AVGSearch, AdamOptimizer

class TestAVGSearch(unittest.TestCase):
    def setUp(self):
        # Use small system for fast testing
        self.search = AVGSearch(n_qubits=4, enterprise_n=1e6)

    def test_initialization(self):
        self.assertEqual(self.search.N, 16)
        self.assertEqual(len(self.search.psi_0), 16)
        self.assertAlmostEqual(np.linalg.norm(self.search.psi_0), 1.0)

    def test_schedule_bounds(self):
        params = np.zeros(4)
        s_0 = self.search.get_schedule(0.0, params)
        s_1 = self.search.get_schedule(1.0, params)
        s_mid = self.search.get_schedule(0.5, params)

        self.assertEqual(s_0, 0.0)
        self.assertEqual(s_1, 1.0)
        self.assertTrue(0.0 <= s_mid <= 1.0)

    def test_simulation_runs(self):
        params = np.zeros(4)
        prob = self.search.simulate_anneal(params, steps=10)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_optimization_improves_result(self):
        # Start with linear schedule (params=0)
        params_start = np.zeros(4)
        loss_start = self.search.loss_function(params_start)

        # Optimize for a few steps
        result = self.search.optimize(iterations=10)
        loss_end = result.losses[-1]

        # Loss should decrease or stay same (unlikely to increase)
        self.assertLessEqual(loss_end, loss_start)

    def test_enterprise_odds_calculation(self):
        result = self.search.optimize(iterations=5)
        self.assertGreater(result.enterprise_odds, 0.0)
        self.assertLess(result.enterprise_odds, 1.0)

    def test_optimizer_step(self):
        params = np.array([0.1, 0.1])
        opt = AdamOptimizer(params, lr=0.1)
        grad = np.array([0.1, -0.1])
        new_params = opt.step(grad)

        self.assertNotEqual(new_params[0], 0.1)
        self.assertNotEqual(new_params[1], 0.1)

if __name__ == '__main__':
    unittest.main()

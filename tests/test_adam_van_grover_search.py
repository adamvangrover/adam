import unittest
import sys
import os

# Bypass core.simulations.__init__ to avoid heavy dependencies for this unit test
sys.path.append(os.path.join(os.getcwd(), 'core', 'simulations'))
import adam_van_grover_search

class TestAdamVanGroverSearch(unittest.TestCase):

    def test_grover_time_calculation(self):
        """Test the Grover time calculation for enterprise scale."""
        N = 1e15
        search_sim = adam_van_grover_search.AdamVanGroverSearch(database_size=N)

        # T_G = (pi/2) * sqrt(10^15) * 1e-3
        # sqrt(10^15) = 31,622,776.6
        # pi/2 = 1.570796
        # product = 49,672,941.3 ns
        # in us = 49,672.94 us

        t_g = search_sim.t_grover
        self.assertAlmostEqual(t_g, 49672.94, delta=10.0)

    def test_single_shot_probability(self):
        """Test the probability calculation against the whitepaper example."""
        N = 1e15
        search_sim = adam_van_grover_search.AdamVanGroverSearch(database_size=N, coherence_time_us=100.0)

        # Whitepaper says approx 2.5e-6 for T_run=50us
        prob = search_sim.calculate_success_probability(run_time_us=50.0)

        # Expected: ~2.5e-6
        # Let's be precise:
        # r = 50 / 49672.94 = 0.00100658
        # theta = r * pi/2 = 0.001581
        # sin(theta)^2 = (0.001581)^2 = 2.499e-6

        self.assertAlmostEqual(prob, 2.5e-6, delta=0.1e-6)

    def test_coherence_limit(self):
        """Test that probability is capped by coherence time."""
        N = 1e15
        # Set coherence to 50us
        search_sim = adam_van_grover_search.AdamVanGroverSearch(database_size=N, coherence_time_us=50.0)

        # Request 100us run time
        prob_100 = search_sim.calculate_success_probability(run_time_us=100.0)
        # Request 50us run time
        prob_50 = search_sim.calculate_success_probability(run_time_us=50.0)

        # Should be equal because 100us is capped at 50us
        self.assertEqual(prob_100, prob_50)

    def test_simulation_batch(self):
        """Test the batch simulation output structure."""
        search_sim = adam_van_grover_search.AdamVanGroverSearch(database_size=1e6) # Smaller scale for test

        result = search_sim.simulate_batch(run_time_us=100.0, num_shots=10)

        self.assertIn("probability_single_shot", result)
        self.assertIn("probability_batch", result)
        self.assertIn("success", result)
        self.assertIn("odds_single_shot", result)
        self.assertEqual(result["num_shots"], 10)

if __name__ == '__main__':
    unittest.main()

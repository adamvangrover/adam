import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation.generator import SimulationGenerator
from core.simulation.sovereign import Sovereign

class TestSimulation(unittest.TestCase):
    def test_generator_initialization(self):
        sim = SimulationGenerator(num_sovereigns=5)
        sim.initialize_world()
        self.assertEqual(len(sim.sovereigns), 5)
        self.assertIsInstance(sim.sovereigns[0], Sovereign)

    def test_simulation_step(self):
        sim = SimulationGenerator(num_sovereigns=5)
        sim.initialize_world()
        sim.run_simulation_step(0)

        # Check that synthetic data was generated
        self.assertTrue(len(sim.synthetic_data) > 0)

        # Check events might be generated (not guaranteed in one step but structure exists)
        self.assertIsInstance(sim.world_events, list)

    def test_full_run(self):
        sim = SimulationGenerator(num_sovereigns=3)
        results = sim.generate_full_simulation(steps=5)

        self.assertIn("sovereigns", results)
        self.assertIn("events", results)
        self.assertIn("synthetic_data", results)
        self.assertEqual(len(results["sovereigns"]), 3)

if __name__ == '__main__':
    unittest.main()

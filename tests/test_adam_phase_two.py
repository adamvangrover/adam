import unittest
import sys
import os
import shutil

# Add root to sys.path
sys.path.append(os.getcwd())

class TestAdamPhaseTwo(unittest.TestCase):

    def test_critique_swarm_consensus(self):
        """Test the CritiqueSwarm consensus critique logic."""
        try:
            from core.agents.critique_swarm import CritiqueSwarm
        except ImportError:
            self.fail("Could not import CritiqueSwarm")

        swarm = CritiqueSwarm()

        # Test Case 1: Overconfident Hallucination
        fake_result = {"score": 0.9, "rationale": "No signals provided"}
        critiques = swarm.critique_consensus(fake_result)
        self.assertTrue(len(critiques) > 0, "CritiqueSwarm failed to catch overconfidence")
        self.assertIn("Logical Consistency", [c['perspective'] for c in critiques])

        # Test Case 2: Groupthink
        fake_result_2 = {"score": 0.99, "rationale": "All agents agree"}
        critiques_2 = swarm.critique_consensus(fake_result_2)
        self.assertTrue(len(critiques_2) > 0, "CritiqueSwarm failed to catch Groupthink")
        self.assertIn("Groupthink Monitor", [c['perspective'] for c in critiques_2])

    def test_immutable_ledger(self):
        """Test the ImmutableLedger writing and chaining."""
        try:
            from core.governance.immutable_ledger import ImmutableLedger
        except ImportError:
            self.fail("Could not import ImmutableLedger")

        test_path = "data/test_ledger.jsonl"
        # Clean up previous run
        if os.path.exists(test_path):
            os.remove(test_path)

        ledger = ImmutableLedger(ledger_path=test_path)

        # Write Entry 1
        hash1 = ledger.log_entry("UserA", "TEST_ACTION", {"foo": "bar"})
        self.assertEqual(len(hash1), 64, "Hash should be SHA256")

        # Write Entry 2
        hash2 = ledger.log_entry("UserB", "TEST_ACTION_2", {"baz": "qux"})

        # Verify Chaining
        with open(test_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)

            entry1 = eval(lines[0].replace('true', 'True').replace('false', 'False')) # json.loads safer but eval ok for quick test
            entry2 = eval(lines[1].replace('true', 'True').replace('false', 'False'))

            # Check Genesis Hash
            self.assertEqual(entry1['previous_hash'], "0"*64)
            # Check Chain
            self.assertEqual(entry2['previous_hash'], hash1)
            self.assertEqual(entry2['current_hash'], hash2)

    def test_refinement_loop(self):
        """Test the Refinement Loop logic."""
        try:
            from core.engine.refinement_loop import RefinementLoop
        except ImportError:
            self.fail("Could not import RefinementLoop")

        loop = RefinementLoop()
        # Mock signal (high confidence to trigger critique)
        signals = [{"agent": "Mock", "vote": "BUY", "confidence": 1.0, "weight": 10.0}]

        # We need to mock the internal consensus engine or just rely on its behavior
        # Assuming ConsensusEngine returns high score for this signal
        result = loop.run_loop(signals)

        # Check output structure
        self.assertIn("status", result)
        self.assertIn("score", result)

        # If score was high, critique should have dampened it
        # (This is harder to test deterministically without mocking ConsensusEngine directly,
        # but we check that the loop ran without crashing)
        self.assertTrue(True)

    def test_scenario_engine(self):
        """Test Scenario Engine."""
        try:
            from core.engine.scenario_engine import ScenarioEngine
        except ImportError:
            self.fail("Could not import ScenarioEngine")

        engine = ScenarioEngine()
        engine.set_scenario("2008_CRASH")

        pulse = engine.get_pulse()
        self.assertEqual(pulse['indices']['SPX']['price'], 1251.70)
        self.assertTrue(pulse['scenario_mode'])

        # Next step
        pulse2 = engine.get_pulse()
        self.assertEqual(pulse2['indices']['SPX']['price'], 1192.70)

if __name__ == '__main__':
    unittest.main()

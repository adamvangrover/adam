from core.utils.config_utils import load_app_config
from core.system.agent_orchestrator import AgentOrchestrator
import unittest
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestV21OrchestratorLoading(unittest.TestCase):

    def test_orchestrator_loads_all_v21_agents(self):
        """
        Tests if the refactored AgentOrchestrator can be instantiated and 
        correctly loads all agents from the new v21 configuration.
        """
        print("\nVerifying Adam v21.0 Orchestrator Agent Loading...")

        try:
            # The orchestrator now loads its own config, so we just need to create it
            orchestrator = AgentOrchestrator()

            # 1. Check that the orchestrator and its agents dictionary exist
            self.assertIsNotNone(orchestrator, "Orchestrator should not be None.")
            self.assertTrue(hasattr(orchestrator, 'agents'), "Orchestrator should have an 'agents' attribute.")
            self.assertIsInstance(orchestrator.agents, dict, "Orchestrator 'agents' should be a dictionary.")

            # 2. Check that agents were actually loaded
            self.assertGreater(len(orchestrator.agents), 0, "The 'agents' dictionary should not be empty.")

            # 3. Load the config manually to get the expected list of agents
            app_config = load_app_config()
            expected_agents = app_config.get("agent_network", {}).get("agent_directory", [])
            expected_agent_names = [agent['name']
                                    for agent in expected_agents if isinstance(agent, dict) and 'name' in agent]

            print(f"Found {len(orchestrator.agents)} loaded agents out of {len(expected_agent_names)} expected agents.")

            # 4. Assert that the number of loaded agents matches the number of configured agents
            self.assertEqual(len(orchestrator.agents), len(expected_agent_names),
                             "Number of loaded agents should match the number in the config file.")

            # 5. Spot-check for the presence of key agents
            self.assertIn("QueryUnderstandingAgent", orchestrator.agents)
            self.assertIn("MetaCognitiveAgent", orchestrator.agents)  # A new v21 agent
            self.assertIn("GeopoliticalRiskAgent", orchestrator.agents)  # A re-added agent
            self.assertIn("industry_specialist_agent_technology", orchestrator.agents)  # A re-added specialized agent

            print("Successfully verified that the AgentOrchestrator loads the new v21 agent configuration.")

        except Exception as e:
            self.fail(f"AgentOrchestrator instantiation or agent loading failed with an exception: {e}")


if __name__ == '__main__':
    unittest.main()

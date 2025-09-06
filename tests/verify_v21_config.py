import unittest
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils.config_utils import load_app_config

class TestV21Config(unittest.TestCase):

    def test_load_v21_configuration(self):
        """
        Tests if the new v21 configuration files can be loaded and parsed without errors.
        """
        print("\nVerifying Adam v21.0 configuration...")
        
        try:
            config = load_app_config()
            self.assertIsNotNone(config, "Loaded configuration should not be None.")
            self.assertIsInstance(config, dict, "Loaded configuration should be a dictionary.")
            
            # Verify that the new agent network structure is present
            self.assertIn("agent_network", config, "The 'agent_network' key should be in the config.")
            self.assertIn("agent_directory", config["agent_network"], "The 'agent_directory' key should be in 'agent_network'.")
            
            # Verify that the new master workflow is present
            self.assertIn("workflows", config, "The 'workflows' key should be in the config.")
            workflow_names = [wf.get("name") for wf in config["workflows"]]
            self.assertIn("adam_v21_master_workflow", workflow_names, "The 'adam_v21_master_workflow' should be defined.")
            
            # Verify that the new ethical framework is present
            self.assertIn("ethical_and_governance_framework", config, "The 'ethical_and_governance_framework' key should be in the config.")
            
            print("Configuration loaded successfully!")
            print("Found 'agent_network' and 'adam_v21_master_workflow'.")
            
        except Exception as e:
            self.fail(f"Configuration loading failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()

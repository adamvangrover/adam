import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAdaptiveAPIReal(unittest.TestCase):

    def test_adaptive_query_initialization(self):
        """
        Verifies that create_app successfully imports and initializes MetaOrchestrator
        without crashing, confirming the file structure and signatures are correct.
        """
        # Mock heavy dependencies
        with patch.dict(sys.modules, {
            'semantic_kernel': MagicMock(),
            'semantic_kernel.connectors.ai.open_ai': MagicMock(),
            'neo4j': MagicMock(),
            'redis': MagicMock(),
            'pika': MagicMock(),
            'spacy': MagicMock(),
            'transformers': MagicMock(),
            'torch': MagicMock(),
            'tensorflow': MagicMock(),
            'dowhy': MagicMock()
        }):
            # Patch internals
            with patch('core.utils.config_utils.load_config') as mock_load_config, \
                 patch('core.system.agent_orchestrator.MessageBroker') as mock_mb, \
                 patch('core.system.agent_orchestrator.LLMPlugin') as mock_llm_plugin:

                # Setup internal mocks
                mock_load_config.return_value = {
                    'agents': {
                        'QueryUnderstandingAgent': {'name': 'QueryUnderstandingAgent'}
                    }
                }
                mock_mb.get_instance.return_value = MagicMock()
                mock_llm_plugin.return_value = MagicMock() # Mock the instance

                # Import api inside the patch context
                try:
                    from services.webapp.api import create_app
                    import services.webapp.api as api_module
                except ImportError as e:
                    self.fail(f"Import failed: {e}")

                # Reset globals to ensure clean state
                api_module.agent_orchestrator = None
                api_module.meta_orchestrator = None

                # Create app
                app = create_app('default')

                # Check if globals are initialized
                if api_module.agent_orchestrator is None:
                     self.fail("AgentOrchestrator is None. Initialization block failed.")

                if api_module.meta_orchestrator is None:
                     self.fail("MetaOrchestrator is None. Initialization block failed.")

                # Verify the type/class name
                self.assertEqual(api_module.agent_orchestrator.__class__.__name__, 'AgentOrchestrator')
                self.assertEqual(api_module.meta_orchestrator.__class__.__name__, 'MetaOrchestrator')

                print("Initialization Successful: MetaOrchestrator is ready.")

                # Test the endpoint routing
                client = app.test_client()

                # Mock route_request
                api_module.meta_orchestrator.route_request = AsyncMock(return_value={"status": "Mock Success"})

                resp = client.post('/api/adaptive/query',
                                   data=json.dumps({"query": "test"}),
                                   content_type='application/json')

                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.get_json(), {"status": "Mock Success"})

if __name__ == '__main__':
    unittest.main()

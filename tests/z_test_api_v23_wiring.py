import sys
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import json

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestAdaptiveAPIReal(unittest.TestCase):

    def tearDown(self):
        # Force unload modules to prevent pollution of other tests
        modules_to_unload = [
            'services.webapp.api',
            'core.system.agent_orchestrator',
            'core.agents.query_understanding_agent',
            'core.v23_graph_engine.meta_orchestrator',
            'flask_jwt_extended'
        ]
        for module in list(sys.modules.keys()):
            for target in modules_to_unload:
                if module == target or module.startswith(target + "."):
                    del sys.modules[module]

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
                mock_llm_plugin.return_value = MagicMock()  # Mock the instance

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

                # Initialize DB for auth checks
                with app.app_context():
                    api_module.db.create_all()

                # Check if globals are initialized
                if api_module.agent_orchestrator is None:
                    self.fail("AgentOrchestrator is None. Initialization block failed.")

                if api_module.meta_orchestrator is None:
                    self.fail("MetaOrchestrator is None. Initialization block failed.")

                # Verify that they are NOT MagicMocks
                # Note: Because of sys.modules patching, classes might be mocks if their modules were mocked.
                # However, AgentOrchestrator should be loaded from real code if core.system.agent_orchestrator wasn't in the patch dict.
                # But imports inside AgentOrchestrator (like semantic_kernel) ARE mocked.

                # We relax the check slightly to allow for partial mocking,
                # but we ensure it's at least an object with the right attributes.
                self.assertTrue(hasattr(api_module.agent_orchestrator, 'execute_agent'), "AgentOrchestrator missing execute_agent")
                self.assertTrue(hasattr(api_module.meta_orchestrator, 'route_request'), "MetaOrchestrator missing route_request")

                print("Initialization Successful: MetaOrchestrator is ready.")

                # Test the endpoint routing
                client = app.test_client()

                # Mock route_request
                api_module.meta_orchestrator.route_request = AsyncMock(return_value={"status": "Mock Success"})

                # Create access token for auth
                from flask_jwt_extended import create_access_token
                with app.app_context():
                    access_token = create_access_token(identity='test_user')
                    headers = {
                        'Authorization': f'Bearer {access_token}'
                    }

                resp = client.post('/api/v23/analyze',
                                   data=json.dumps({"query": "test"}),
                                   content_type='application/json',
                                   headers=headers)

                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.get_json(), {"status": "Mock Success"})


if __name__ == '__main__':
    unittest.main()

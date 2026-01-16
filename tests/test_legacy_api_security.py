import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import importlib.util
import os

# Mock AgentOrchestrator and Echo to avoid initialization issues during import
sys.modules['core.system.agent_orchestrator'] = MagicMock()
sys.modules['core.system.echo'] = MagicMock()

# Import core/api.py by path because it is shadowed by core/api/ package
file_path = os.path.join(os.getcwd(), "core/api.py")
spec = importlib.util.spec_from_file_location("legacy_api", file_path)
legacy_api = importlib.util.module_from_spec(spec)
sys.modules["legacy_api"] = legacy_api
spec.loader.exec_module(legacy_api)

from legacy_api import app

class TestCoreApiSecurity(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_error_leakage(self):
        # Arrange
        error_message = "Sensitive internal detail: /var/www/html/secret.key"

        with patch.object(legacy_api, 'get_knowledge_graph_data', side_effect=Exception(error_message)):
            payload = {
                "module": "knowledge_graph",
                "action": "get_data",
                "parameters": {"module": "test", "concept": "test"}
            }
            headers = {"X-API-Key": "default-insecure-key-change-me"}

            # Act
            response = self.app.post('/', json=payload, headers=headers)

            # Assert
            data = json.loads(response.data)

            # Verify the vulnerability IS FIXED
            self.assertIn("error", data)
            self.assertEqual(data["error"], "An internal error occurred.")  # Generic message
            self.assertNotEqual(data["error"], error_message)  # NO leak
            self.assertEqual(response.status_code, 500)
            print("FIX CONFIRMED: Error message is generic.")

    def test_unauthorized_access(self):
        """Test that requests without a valid API key are rejected."""
        payload = {
            "module": "knowledge_graph",
            "action": "get_data",
            "parameters": {"module": "test", "concept": "test"}
        }

        # Case 1: No API Key
        response = self.app.post('/', json=payload)
        self.assertEqual(response.status_code, 401)
        self.assertIn("Unauthorized", response.json["error"])

        # Case 2: Invalid API Key
        headers = {"X-API-Key": "wrong-key"}
        response = self.app.post('/', json=payload, headers=headers)
        self.assertEqual(response.status_code, 401)
        self.assertIn("Unauthorized", response.json["error"])


if __name__ == '__main__':
    unittest.main()

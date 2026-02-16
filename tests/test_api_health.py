import sys
from unittest.mock import MagicMock
from flask import Blueprint

# Mock dependencies to avoid ImportErrors
mock_bp_module = MagicMock()
mock_bp = Blueprint('quantum', __name__)
mock_bp_module.quantum_bp = mock_bp
sys.modules['services.webapp.blueprints.quantum_blueprint'] = mock_bp_module

sys.modules['semantic_kernel'] = MagicMock()
sys.modules['core.agents.snc_analyst_agent'] = MagicMock()
sys.modules['core.simulations'] = MagicMock()
sys.modules['core.simulations.Credit_Rating_Assessment_Simulation'] = MagicMock()
sys.modules['core.agents.specialized.quantum_retrieval_agent'] = MagicMock()

import unittest
import os
import json

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.webapp.api import create_app, db

class TestAPIHealth(unittest.TestCase):
    """
    🛡️ Sentinel Regression Test: Verify API Health.
    Ensures that basic API endpoints are still accessible after security changes.
    """

    def setUp(self):
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_hello_world(self):
        """
        Test that /api/hello returns 200 OK.
        """
        response = self.client.get('/api/hello')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Hello, World!')

if __name__ == '__main__':
    unittest.main()

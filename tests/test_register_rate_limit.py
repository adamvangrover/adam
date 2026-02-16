import sys
from unittest.mock import MagicMock
from flask import Blueprint

# Mock the blueprint module to avoid deep dependency tree
mock_bp_module = MagicMock()
mock_bp = Blueprint('quantum', __name__)
mock_bp_module.quantum_bp = mock_bp
sys.modules['services.webapp.blueprints.quantum_blueprint'] = mock_bp_module

# Mock semantic_kernel and other core modules to prevent ImportErrors
sys.modules['semantic_kernel'] = MagicMock()
sys.modules['core.agents.snc_analyst_agent'] = MagicMock()
sys.modules['core.simulations'] = MagicMock()
sys.modules['core.simulations.Credit_Rating_Assessment_Simulation'] = MagicMock()
sys.modules['core.agents.specialized.quantum_retrieval_agent'] = MagicMock()

import unittest
import os
import json
import time

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.webapp.api import create_app, db, User

class TestRegisterRateLimit(unittest.TestCase):
    """
    🛡️ Sentinel Security Test: Verify Rate Limiting on Registration.
    Ensures that the application rejects excessive registration attempts from the same IP.
    """

    def setUp(self):
        # Use testing config which uses in-memory DB
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_register_rate_limit(self):
        """
        Test that making more than 5 registration requests in a minute returns 429.
        """

        for i in range(5):
            username = f'user_test_{i}'
            response = self.client.post('/api/register', json={
                'username': username,
                'password': 'SecurePass123!@#'
            })
            # We don't strictly care if it succeeds (201) or fails (400),
            # just that it's NOT 429 yet.
            self.assertNotEqual(response.status_code, 429, f"Request {i+1} shouldn't be rate limited. Got {response.status_code}")

        # The 6th attempt should fail with 429
        response = self.client.post('/api/register', json={
            'username': 'user_limit_test',
            'password': 'SecurePass123!@#'
        })

        self.assertEqual(
            response.status_code,
            429,
            f"6th registration attempt should be rate limited (429). Got {response.status_code}"
        )

if __name__ == '__main__':
    unittest.main()

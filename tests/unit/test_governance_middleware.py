import unittest
from flask import Flask
from unittest import mock

from services.webapp import governance as governance_module
from services.webapp.governance import GovernanceMiddleware
import json
import logging
import os

class TestGovernanceMiddleware(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        # Ensure we use the correct path relative to repo root
        policy_path = 'config/governance_policy.yaml'
        if not os.path.exists(policy_path):
            # Fallback for running tests from inside 'tests/' directory
            policy_path = '../config/governance_policy.yaml'

        self.governance = GovernanceMiddleware(self.app, policy_path=policy_path)

        @self.app.route('/api/trade', methods=['POST'])
        def trade():
            return "Trade executed"

        @self.app.route('/api/public', methods=['GET'])
        def public():
            return "Public data"

        @self.app.route('/api/post_data', methods=['POST'])
        def post_data():
            return "Data posted"

        self.client = self.app.test_client()

    def test_allowed_request(self):
        response = self.client.get('/api/public')
        self.assertEqual(response.status_code, 200)

    def test_restricted_request(self):
        # /api/trade POST is HIGH RISK.
        # Protocol: ADAM-V-NEXT - Strict Enforcement check.
        # Should now return 403 Forbidden.

        # Patch the module's own logging reference: other tests may reconfigure or replace root logging.
        with mock.patch.object(governance_module.logging, "warning") as warn:
            response = self.client.post('/api/trade')
        self.assertEqual(response.status_code, 403)
        self.assertTrue(any("Governance Alert" in str(c.args[0]) for c in warn.call_args_list))
        self.assertIn("High risk operation blocked", response.get_data(as_text=True))

    def test_blacklisted_keyword(self):
        # Use data=... which sends form data or body
        response = self.client.post('/api/post_data', data="Please DROP TABLE users")
        self.assertEqual(response.status_code, 400)
        self.assertIn("blocked by governance policy", response.get_data(as_text=True))

if __name__ == '__main__':
    unittest.main()

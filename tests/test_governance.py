import unittest
from flask import Flask
from services.webapp.governance import GovernanceMiddleware
import json
import logging

class TestGovernance(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.governance = GovernanceMiddleware(self.app, policy_path='config/governance_policy.yaml')

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

        with self.assertLogs() as captured:
            response = self.client.post('/api/trade')
            self.assertEqual(response.status_code, 403)
            self.assertIn("Governance Block", response.get_data(as_text=True))

    def test_blacklisted_keyword(self):
        response = self.client.post('/api/post_data', data="Please DROP TABLE users")
        self.assertEqual(response.status_code, 400)
        self.assertIn("blocked by governance policy", response.get_data(as_text=True))

if __name__ == '__main__':
    unittest.main()

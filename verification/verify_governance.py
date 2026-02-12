import sys
import os
import json
import unittest
import logging

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

from services.webapp.api import create_app, db, User

class TestGovernance(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

        # Create Test User
        if not User.query.filter_by(username='testuser').first():
            u = User(username='testuser')
            u.set_password('Password123!')
            db.session.add(u)
            db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def get_auth_headers(self):
        res = self.client.post('/api/login', json={'username': 'testuser', 'password': 'Password123!'})
        data = json.loads(res.data)
        token = data['access_token']
        return {'Authorization': f'Bearer {token}'}

    def test_restricted_endpoint_block(self):
        # /api/admin/audit_logs is restricted to ADMIN role.
        # testuser has default role 'user'.
        # However, governance middleware might block it based on path too.
        # config/governance_policy.yaml says:
        # - path: "/api/admin" -> risk_level: "CRITICAL" -> blocked?

        # Actually GovernanceMiddleware blocks HIGH/CRITICAL unless override token is present.
        headers = self.get_auth_headers()

        # This should be blocked by middleware (403)
        res = self.client.get('/api/admin/audit_logs', headers=headers)

        # It might return 403 from @require_permission or from GovernanceMiddleware.
        # GovernanceMiddleware runs before request.

        print(f"\n[TEST] Accessing Restricted Endpoint: {res.status_code}")
        self.assertEqual(res.status_code, 403)
        self.assertIn("Access denied", res.get_data(as_text=True))

    def test_safe_endpoint_allow(self):
        headers = self.get_auth_headers()
        res = self.client.get('/api/hello', headers=headers)
        self.assertEqual(res.status_code, 200)
        print(f"\n[PASS] Safe Endpoint Allowed.")

if __name__ == '__main__':
    unittest.main()

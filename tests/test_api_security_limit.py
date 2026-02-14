import unittest
import os
import sys
import json

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.webapp.api import create_app, db, User

class TestAPISecurityLimit(unittest.TestCase):
    """
    🛡️ Sentinel Security Test: Verify DoS Protection.
    Ensures that the application rejects payloads larger than the configured limit (16MB).
    """

    def setUp(self):
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

        # Create a test user
        self.user = User(username='sentinel_test')
        self.user.set_password('SecurePass123!')
        db.session.add(self.user)
        db.session.commit()

        # Login to get token
        resp = self.client.post('/api/login', json={
            'username': 'sentinel_test',
            'password': 'SecurePass123!'
        })
        self.token = resp.get_json()['access_token']
        self.headers = {'Authorization': f'Bearer {self.token}'}

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_max_content_length_enforcement(self):
        """
        Test that a request larger than MAX_CONTENT_LENGTH (16MB) is rejected with 413.
        """
        # 17MB payload (17 * 1024 * 1024 bytes)
        # This exceeds the 16MB limit we expect to enforce.
        large_payload = 'a' * (17 * 1024 * 1024)

        # Use JSON to bypass Werkzeug's default form-data limits and hit the global limit
        response = self.client.post('/api/v23/crisis_response',
                                  json={'prompt': large_payload},
                                  headers=self.headers)

        self.assertEqual(
            response.status_code,
            413,
            "Server should reject payloads > 16MB with 413 Payload Too Large"
        )

if __name__ == '__main__':
    unittest.main()

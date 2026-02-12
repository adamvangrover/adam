import sys
import os
import json
import unittest
import logging

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress logging during tests
logging.getLogger('werkzeug').setLevel(logging.ERROR)

from services.webapp.api import create_app, db, User

class TestApiEndpoints(unittest.TestCase):
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

    def test_synthesizer_confidence(self):
        headers = self.get_auth_headers()
        res = self.client.get('/api/synthesizer/confidence', headers=headers)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('score', data)
        self.assertIn('consensus', data)
        print(f"\n[PASS] Confidence Score: {data['score']}")

    def test_synthesizer_narratives(self):
        headers = self.get_auth_headers()
        res = self.client.get('/api/synthesizer/narratives', headers=headers)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIsInstance(data, list)
        if len(data) > 0:
            self.assertIn('theme', data[0])
            self.assertIn('sentiment', data[0])
            print(f"\n[PASS] Retrieved {len(data)} narratives.")
            print(f"      Example: {data[0]['headline']}")
        else:
            print("\n[WARN] No narratives returned (Mock fallback empty?)")

    def test_intercom_stream(self):
        headers = self.get_auth_headers()
        res = self.client.get('/api/intercom/stream', headers=headers)
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIsInstance(data, list)
        print(f"\n[PASS] Intercom Stream: {len(data)} thoughts retrieved.")

if __name__ == '__main__':
    unittest.main()

import json
import unittest

from services.webapp.api import User, create_app, db


class SimulationSecurityTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        self.client = self.app.test_client()

        # Create user and login
        user = User(username='testuser')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        response = self.client.post('/api/login',
                                 data=json.dumps({'username': 'testuser', 'password': 'password'}),
                                 content_type='application/json')
        self.access_token = json.loads(response.data)['access_token']
        self.headers = {'Authorization': f'Bearer {self.access_token}'}

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_run_valid_simulation(self):
        # We assume Credit_Rating_Assessment_Simulation exists
        # It might return 200 (if queued) or 500 (if celery connection fails)
        # But it MUST NOT return 400 (Invalid simulation name)
        response = self.client.post('/api/simulations/Credit_Rating_Assessment_Simulation',
                                  headers=self.headers)
        self.assertNotEqual(response.status_code, 400)

    def test_run_invalid_simulation(self):
        # This currently returns 200 (Vulnerability) or 500, but we want 400 (Fixed)
        response = self.client.post('/api/simulations/Malicious_Simulation',
                                  headers=self.headers)
        self.assertEqual(response.status_code, 400)

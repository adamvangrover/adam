import unittest
import json
from unittest.mock import patch
from .api import create_app, db, User

class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_hello(self):
        response = self.client.get('/api/hello')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Hello, World!')

    def test_get_agents(self):
        response = self.client.get('/api/agents')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, [])


    def test_login(self):
        # Create a test user
        user = User(username='testuser')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()

        # Test successful login
        response = self.client.post('/api/login',
                                 data=json.dumps({'username': 'testuser', 'password': 'password'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('access_token', data)

        # Test failed login
        response = self.client.post('/api/login',
                                 data=json.dumps({'username': 'testuser', 'password': 'wrongpassword'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertIn('error', data)

    @patch('services.webapp.api.agent_orchestrator')
    def test_invoke_agent(self, mock_agent_orchestrator):
        mock_agent_orchestrator.run_agent.return_value = {'result': 'success'}
        response = self.client.post('/api/agents/some_agent/invoke',
                                 data=json.dumps({'some': 'data'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, {'result': 'success'})
        mock_agent_orchestrator.run_agent.assert_called_with('some_agent', {'some': 'data'})


if __name__ == '__main__':
    unittest.main()

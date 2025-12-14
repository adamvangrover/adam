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
        mock_agent_orchestrator.execute_agent.return_value = {'result': 'success'}
        response = self.client.post('/api/agents/some_agent/invoke',
                                 data=json.dumps({'some': 'data'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data, {'result': 'success'})
        mock_agent_orchestrator.execute_agent.assert_called_with('some_agent', {'some': 'data'})

    def test_portfolio_endpoints(self):
        # Create a test user and login
        user = User(username='testuser', role='user')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        response = self.client.post('/api/login',
                                 data=json.dumps({'username': 'testuser', 'password': 'password'}),
                                 content_type='application/json')
        access_token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}

        # Create a portfolio
        response = self.client.post('/api/portfolios',
                                  headers=headers,
                                  data=json.dumps({'name': 'My Test Portfolio'}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        portfolio_id = json.loads(response.data)['id']

        # Add an asset
        response = self.client.post(f'/api/portfolios/{portfolio_id}/assets',
                                  headers=headers,
                                  data=json.dumps({'symbol': 'AAPL', 'quantity': 10, 'purchase_price': 150.0}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 200)
        asset_id = json.loads(response.data)['id']

        # Get the portfolio and check the asset
        response = self.client.get(f'/api/portfolios/{portfolio_id}', headers=headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(len(data['assets']), 1)
        self.assertEqual(data['assets'][0]['symbol'], 'AAPL')

        # Update the asset
        response = self.client.put(f'/api/portfolios/{portfolio_id}/assets/{asset_id}',
                                 headers=headers,
                                 data=json.dumps({'symbol': 'AAPL', 'quantity': 15, 'purchase_price': 155.0}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['quantity'], 15)

        # Delete the asset
        response = self.client.delete(f'/api/portfolios/{portfolio_id}/assets/{asset_id}', headers=headers)
        self.assertEqual(response.status_code, 200)

        # Delete the portfolio
        response = self.client.delete(f'/api/portfolios/{portfolio_id}', headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_security_headers(self):
        response = self.client.get('/api/hello')
        headers = response.headers
        self.assertIn('X-Content-Type-Options', headers)
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn('X-Frame-Options', headers)
        self.assertEqual(headers['X-Frame-Options'], 'SAMEORIGIN')
        self.assertIn('Strict-Transport-Security', headers)
        self.assertIn('max-age=31536000', headers['Strict-Transport-Security'])


if __name__ == '__main__':
    unittest.main()

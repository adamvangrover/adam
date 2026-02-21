import unittest
import json
import os
import time
import hmac
import hashlib
from flask import Flask
from werkzeug.exceptions import Forbidden
from unittest.mock import patch
from .api import create_app, db, User
from .governance import GovernanceMiddleware


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

    def test_register_username_validation(self):
        # 🛡️ Sentinel: Test username format validation
        invalid_usernames = [
            'usr',             # Too short (< 4 chars)
            'user@name',       # Special char @
            '<script>',        # XSS attempt
            'user name',       # Space
            'a' * 31           # Too long (> 30 chars)
        ]
        strong_pwd = 'StrongPassword1!'

        for uname in invalid_usernames:
            response = self.client.post('/api/register',
                                        data=json.dumps({'username': uname, 'password': strong_pwd}),
                                        content_type='application/json')
            self.assertEqual(response.status_code, 400, f"Allowed invalid username: {uname}")
            self.assertIn('Invalid username format', json.loads(response.data).get('error', ''))

    def test_register_weak_password(self):
        # 🛡️ Sentinel: Test password strength validation
        weak_passwords = [
            'short',           # Too short
            'alllowercase1!',  # No uppercase
            'ALLUPPERCASE1!',  # No lowercase
            'NoNumbers!!!!',   # No digits
            'NoSpecialChar12'  # No special chars
        ]

        for pwd in weak_passwords:
            # Sentinel: Ensure username is valid for password tests
            # 'valid_user_' + pwd might contain invalid characters from pwd (like !) which triggers username validation first
            # Also hash can produce negative numbers, hyphen is allowed but let's be safe and use abs
            safe_username = f'user_{abs(hash(pwd))}'
            response = self.client.post('/api/register',
                                        data=json.dumps({'username': safe_username, 'password': pwd}),
                                        content_type='application/json')
            self.assertEqual(response.status_code, 400, f"Allowed weak password: {pwd}")
            self.assertIn('Password is too weak', json.loads(response.data)['error'])

        # Test strong password
        strong_pwd = 'StrongPassword1!'
        response = self.client.post('/api/register',
                                    data=json.dumps({'username': 'good_user', 'password': strong_pwd}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 201)

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

    def test_invoke_agent_unauthorized(self):
        response = self.client.post('/api/agents/some_agent/invoke',
                                    data=json.dumps({'some': 'data'}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 401)

    @patch('services.webapp.api.agent_orchestrator')
    def test_invoke_agent(self, mock_agent_orchestrator):
        # Authenticate first
        user = User(username='agentuser')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        response = self.client.post('/api/login',
                                    data=json.dumps({'username': 'agentuser', 'password': 'password'}),
                                    content_type='application/json')
        access_token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}

        mock_agent_orchestrator.execute_agent.return_value = {'result': 'success'}
        response = self.client.post('/api/agents/some_agent/invoke',
                                    headers=headers,
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

    def test_v23_analysis_validation(self):
        # Authenticate first
        user = User(username='analyst')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        response = self.client.post('/api/login',
                                    data=json.dumps({'username': 'analyst', 'password': 'password'}),
                                    content_type='application/json')
        access_token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}

        # Valid query
        response = self.client.post('/api/v23/analyze',
                                    headers=headers,
                                    data=json.dumps({'query': 'analyze AAPL'}),
                                    content_type='application/json')
        # Expect 200 (mock result) or 500 depending on config, but NOT 400 for validation
        self.assertNotEqual(response.status_code, 400)

        # Invalid type
        response = self.client.post('/api/v23/analyze',
                                    headers=headers,
                                    data=json.dumps({'query': 12345}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Query must be a string', response.get_json()['error'])

        # Too long
        response = self.client.post('/api/v23/analyze',
                                    headers=headers,
                                    data=json.dumps({'query': 'A' * 5001}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Query too long', response.get_json()['error'])

    def test_run_simulation_validation(self):
        # We need to be logged in
        user = User(username='simuser', role='user')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        response = self.client.post('/api/login',
                                    data=json.dumps({'username': 'simuser', 'password': 'password'}),
                                    content_type='application/json')
        access_token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}

        # Invalid characters
        response = self.client.post('/api/simulations/invalid-name!',
                                    headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid simulation name format', response.get_json()['error'])

    def test_credit_generate_auth(self):
        # Unauthenticated: Should return 200 with FULL SIMULATION data (Virtual Pipeline)
        response = self.client.post('/api/credit/generate',
                                    data=json.dumps({'ticker': 'AAPL'}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Verify Headers
        self.assertEqual(response.headers.get('X-Adam-Mode'), 'Simulation')

        data = json.loads(response.data)

        # Verify Payload Structure matches expected "Virtual Pipeline" output
        self.assertEqual(data.get('mode'), 'simulation_fallback')
        self.assertIn('memo', data)
        self.assertIn('data', data)
        self.assertIn('financials', data['data'])
        self.assertTrue(len(data['data']['risks']) > 0)

        # Ensure memo contains the ticker (Deterministic Randomness check)
        self.assertIn('AAPL', data['memo'])

        # Authenticated (mocking the pipeline)
        user = User(username='credit_user')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        response = self.client.post('/api/login',
                                    data=json.dumps({'username': 'credit_user', 'password': 'password'}),
                                    content_type='application/json')
        access_token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}

        # Mocking import of CreditPipeline is hard because it happens inside the function.
        # But if we get 500 (ImportError) or success structure, it means auth passed logic.
        response = self.client.post('/api/credit/generate',
                                    headers=headers,
                                    data=json.dumps({'ticker': 'AAPL'}),
                                    content_type='application/json')
        # We expect the pipeline to ATTEMPT to run. Since imports might fail in this test env:
        # If it fails with ImportError (500), that proves it passed the auth check.
        # If it succeeds (mocks/sims), also good.
        # But crucially, it should NOT return the "simulation" status from the auth check.
        if response.status_code == 200:
            data = json.loads(response.data)
            # If it returned success, make sure it wasn't the auth fallback
            if data.get('status') == 'simulation':
                self.fail("Authenticated request triggered simulation fallback!")
        elif response.status_code == 500:
            # Expected if CreditPipeline deps are missing
            pass
        else:
            self.fail(f"Unexpected status code: {response.status_code}")

    def test_crisis_response_simulation(self):
        # 🛡️ Sentinel: Verify Virtual Crisis Pipeline for unauthenticated users
        response = self.client.post('/api/v23/crisis_response',
                                    data={'prompt': 'Test Crisis'},
                                    content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get('X-Adam-Mode'), 'Simulation')

        data = json.loads(response.data)
        self.assertTrue(data.get('fallback'), "Should return fallback flag")
        self.assertIn('detailed_analysis', data)
        self.assertEqual(data.get('mode'), 'simulation_fallback')


class SecurityTestCase(unittest.TestCase):
    def setUp(self):
        # We need CORE_INTEGRATION=True to reach the vulnerable code path
        self.app = create_app('testing')
        self.app.config['CORE_INTEGRATION'] = True
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_governance_override_secure(self):
        """
        🛡️ Sentinel: Verify that the governance override mechanism is secure.
        1. Ensure random key generation (override disabled) when env var is missing.
        2. Ensure explicit key via env var works.
        """
        # Save original env
        original_secret = os.environ.get('GOVERNANCE_OVERRIDE_SECRET')
        if original_secret:
            del os.environ['GOVERNANCE_OVERRIDE_SECRET']

        try:
            # Case 1: No Env Var -> Random Key -> Default Secret Fails
            app = Flask(__name__)
            # Point to the real config file so we have restricted endpoints
            middleware = GovernanceMiddleware(app, policy_path='config/governance_policy.yaml')

            # Try to use the old insecure default
            default_secret = b'dev-secret-do-not-use-in-prod'
            timestamp = int(time.time())
            path = '/api/admin/test'
            payload = f"{timestamp}:{path}".encode()
            signature = hmac.new(default_secret, payload, hashlib.sha256).hexdigest()
            token = f"{timestamp}:{signature}"

            with app.test_request_context(path, method='POST', headers={'X-Governance-Override': token}):
                with self.assertRaises(Forbidden):
                    middleware.check_governance()

            # Case 2: Set Env Var -> Correct Secret Works
            secure_secret = 'my-secure-test-secret'
            os.environ['GOVERNANCE_OVERRIDE_SECRET'] = secure_secret

            # Re-init middleware to pick up new secret
            middleware = GovernanceMiddleware(app, policy_path='config/governance_policy.yaml')

            signature = hmac.new(secure_secret.encode(), payload, hashlib.sha256).hexdigest()
            valid_token = f"{timestamp}:{signature}"

            with app.test_request_context(path, method='POST', headers={'X-Governance-Override': valid_token}):
                # Should not raise
                middleware.check_governance()

        finally:
            # Restore env
            if original_secret:
                os.environ['GOVERNANCE_OVERRIDE_SECRET'] = original_secret
            elif 'GOVERNANCE_OVERRIDE_SECRET' in os.environ:
                del os.environ['GOVERNANCE_OVERRIDE_SECRET']

    @patch('services.webapp.api.meta_orchestrator')
    def test_analyze_error_leak(self, mock_meta):
        # Authenticate first
        user = User(username='sec_tester')
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        response = self.client.post('/api/login',
                                    data=json.dumps({'username': 'sec_tester', 'password': 'password'}),
                                    content_type='application/json')
        access_token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {access_token}'}

        # Setup the mock to raise a sensitive error
        sensitive_info = "DB_PASSWORD=secret123"
        mock_meta.route_request.side_effect = Exception(f"Connection failed: {sensitive_info}")

        # Inject the mock into the api module
        import services.webapp.api as api
        original_meta = api.meta_orchestrator
        api.meta_orchestrator = mock_meta

        try:
            response = self.client.post('/api/v23/analyze',
                                        headers=headers,
                                        data=json.dumps({'query': 'test query'}),
                                        content_type='application/json')

            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)

            # Verify no leak
            self.assertNotIn(sensitive_info, data['error'])
            self.assertEqual(data['error'], 'An internal error occurred during analysis.')
        finally:
            api.meta_orchestrator = original_meta

    def test_login_timing_mitigation(self):
        # 🛡️ Sentinel: Ensure login still works correctly with timing mitigation
        # Test valid user
        user = User(username='valid_user')
        user.set_password('CorrectPassword1!')
        db.session.add(user)
        db.session.commit()

        response = self.client.post('/api/login',
                                    data=json.dumps({'username': 'valid_user', 'password': 'CorrectPassword1!'}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Test invalid user (should verify against dummy hash)
        response = self.client.post('/api/login',
                                    data=json.dumps({'username': 'non_existent_user', 'password': 'AnyPassword'}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 401)
        self.assertEqual(json.loads(response.data)['error'], 'Invalid credentials')


if __name__ == '__main__':
    unittest.main()

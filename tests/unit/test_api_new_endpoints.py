import unittest
from services.webapp.api import create_app, db
from flask_jwt_extended import create_access_token

class TestAPIIntegration(unittest.TestCase):
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

    def test_synthesizer_endpoint(self):
        # Create a token
        with self.client:
             access_token = create_access_token(identity="1")
             headers = {'Authorization': f'Bearer {access_token}'}

             response = self.client.get('/api/synthesizer/confidence', headers=headers)
             if response.status_code != 200:
                 print(f"Error 1: {response.get_data(as_text=True)}")
             self.assertEqual(response.status_code, 200)
             data = response.get_json()
             print("\nSynthesizer Data:", data)
             self.assertIn('score', data)
             self.assertIn('pulse', data)

    def test_intercom_endpoint(self):
        with self.client:
             access_token = create_access_token(identity="1")
             headers = {'Authorization': f'Bearer {access_token}'}

             response = self.client.get('/api/intercom/stream', headers=headers)
             if response.status_code != 200:
                 print(f"Error 2: {response.get_data(as_text=True)}")
             self.assertEqual(response.status_code, 200)
             data = response.get_json()
             print("\nIntercom Data:", data)
             self.assertTrue(len(data) > 0)

             # Bolt Validation: Ensure response contains objects with IDs
             first_thought = data[0]
             self.assertIsInstance(first_thought, dict)
             self.assertIn('id', first_thought)
             self.assertIn('text', first_thought)

if __name__ == '__main__':
    unittest.main()

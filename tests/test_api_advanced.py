import unittest
from services.webapp.api import create_app, db
from flask_jwt_extended import create_access_token

class TestAPIAdvancedEndpoints(unittest.TestCase):
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

    def test_forecast_endpoint(self):
        with self.client:
             access_token = create_access_token(identity="1")
             headers = {'Authorization': f'Bearer {access_token}'}

             # Test SPX forecast
             response = self.client.get('/api/synthesizer/forecast/SPX', headers=headers)
             if response.status_code != 200:
                 print("Error Forecast:", response.get_data(as_text=True))

             self.assertEqual(response.status_code, 200)
             data = response.get_json()
             self.assertIn('history', data)
             self.assertIn('forecast', data)
             self.assertIn('upper_95', data['forecast'])

    def test_conviction_endpoint(self):
        with self.client:
             access_token = create_access_token(identity="1")
             headers = {'Authorization': f'Bearer {access_token}'}

             response = self.client.get('/api/synthesizer/conviction', headers=headers)
             self.assertEqual(response.status_code, 200)
             data = response.get_json()
             print("Conviction:", data)
             self.assertIn('scores', data)
             self.assertIn('RiskOfficer', data['scores'])

if __name__ == '__main__':
    unittest.main()

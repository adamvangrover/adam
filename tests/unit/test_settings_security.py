import unittest
from pydantic import ValidationError
from core.settings import Settings
import os

class TestSettingsSecurity(unittest.TestCase):
    def test_default_key_is_secure(self):
        """Test that by default, a secure random key is generated."""
        # Ensure environment is clean
        if "ADAM_API_KEY" in os.environ:
            del os.environ["ADAM_API_KEY"]

        settings_1 = Settings(environment="development")
        settings_2 = Settings(environment="development")

        self.assertNotEqual(settings_1.adam_api_key, "default-insecure-key-change-me")
        self.assertTrue(len(settings_1.adam_api_key) > 32)
        # Should generate a new key each time if not provided
        self.assertNotEqual(settings_1.adam_api_key, settings_2.adam_api_key)

    def test_production_mode_allows_secure_key(self):
        """Test that production mode allows a custom secure key."""
        secure_key = "secure-random-key-123"
        settings = Settings(environment="production", adam_api_key=secure_key)
        self.assertEqual(settings.adam_api_key, secure_key)
        self.assertEqual(settings.environment, "production")

if __name__ == '__main__':
    unittest.main()

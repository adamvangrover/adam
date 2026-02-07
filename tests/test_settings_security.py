import unittest
from pydantic import ValidationError
from core.settings import Settings
import os

class TestSettingsSecurity(unittest.TestCase):
    def test_development_mode_allows_default_key(self):
        """Test that development mode allows the default insecure key (with warning)."""
        # Ensure environment is clean
        if "ADAM_API_KEY" in os.environ:
            del os.environ["ADAM_API_KEY"]

        settings = Settings(environment="development")
        self.assertEqual(settings.adam_api_key, "default-insecure-key-change-me")
        self.assertEqual(settings.environment, "development")

    def test_production_mode_blocks_default_key(self):
        """Test that production mode raises ValueError if default key is used."""
        # Ensure environment is clean
        if "ADAM_API_KEY" in os.environ:
            del os.environ["ADAM_API_KEY"]

        with self.assertRaises(ValueError) as cm:
            Settings(environment="production")

        self.assertIn("CRITICAL SECURITY ERROR", str(cm.exception))

    def test_production_mode_allows_secure_key(self):
        """Test that production mode allows a custom secure key."""
        secure_key = "secure-random-key-123"
        settings = Settings(environment="production", adam_api_key=secure_key)
        self.assertEqual(settings.adam_api_key, secure_key)
        self.assertEqual(settings.environment, "production")

if __name__ == '__main__':
    unittest.main()

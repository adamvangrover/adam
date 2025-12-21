import unittest
import os
import logging
from unittest.mock import patch

# Add the project root to the Python path to allow direct import of core modules
# This might be needed if running the test file directly and the test runner doesn't handle it.
# However, for a structured project, it's better if PYTHONPATH is set correctly
# or the test runner (e.g., 'python -m unittest discover') handles discovery.
# For now, assuming the execution environment can find 'core'.
from core.utils.secrets_utils import get_api_key


class TestSecretsUtils(unittest.TestCase):

    @patch.dict(os.environ, {'TEST_API_KEY_EXISTS': 'test_value_123'}, clear=True)
    def test_get_api_key_exists(self):
        """Test that get_api_key returns the value if the environment variable exists."""
        self.assertEqual(get_api_key('TEST_API_KEY_EXISTS'), 'test_value_123')

    @patch.dict(os.environ, {}, clear=True)  # Ensure the key is not set
    def test_get_api_key_not_exists(self):
        """Test that get_api_key returns None and logs a warning if the key does not exist."""
        # Ensure 'TEST_API_KEY_NOT_EXISTS' is not in os.environ (covered by patch.dict with clear=True)
        # The logger in secrets_utils.py uses logging.warning, which goes to the root logger.
        # We need to capture logs from the logger where secrets_utils.py issues them.
        # If secrets_utils.py does `logging.getLogger(__name__)`, then it's 'core.utils.secrets_utils'.
        # If it does `logging.warning()`, it's the root logger.
        # The current implementation of secrets_utils.py uses `logging.warning`,
        # so we capture from the root logger.
        with self.assertLogs(logger=None, level='WARNING') as cm:  # logger=None captures root logger
            result = get_api_key('TEST_API_KEY_NOT_EXISTS')

        self.assertIsNone(result)
        self.assertIn("API key 'TEST_API_KEY_NOT_EXISTS' not found in environment variables.", cm.output[0])

    @patch.dict(os.environ, {'TEST_API_KEY_EMPTY': ''}, clear=True)
    def test_get_api_key_empty_value(self):
        """Test that get_api_key returns None and logs a warning for an empty string value."""
        with self.assertLogs(logger=None, level='WARNING') as cm:
            result = get_api_key('TEST_API_KEY_EMPTY')

        self.assertIsNone(result)
        self.assertIn(
            "API key 'TEST_API_KEY_EMPTY' found in environment variables but is empty or whitespace.", cm.output[0])

    @patch.dict(os.environ, {'TEST_API_KEY_WHITESPACE': '   '}, clear=True)
    def test_get_api_key_whitespace_value(self):
        """Test that get_api_key returns None and logs a warning for a whitespace-only string value."""
        with self.assertLogs(logger=None, level='WARNING') as cm:
            result = get_api_key('TEST_API_KEY_WHITESPACE')

        self.assertIsNone(result)
        self.assertIn(
            "API key 'TEST_API_KEY_WHITESPACE' found in environment variables but is empty or whitespace.", cm.output[0])


if __name__ == '__main__':
    unittest.main()

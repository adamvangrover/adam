# tests/test_config_utils.py

import unittest
import tempfile
import os
import yaml
from core.utils.config_utils import load_config, load_error_codes
from core.system.error_handler import ConfigurationError

class TestConfigUtils(unittest.TestCase):

    def test_load_config_success(self):
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmpfile:
            yaml.dump({"key": "value"}, tmpfile)
            tmpfile_path = tmpfile.name

        config = load_config(tmpfile_path)
        self.assertEqual(config, {"key": "value"})
        os.remove(tmpfile_path)  # Clean up

    def test_load_config_file_not_found(self):
        config = load_config("nonexistent_file.yaml")
        self.assertIsNone(config)

    def test_load_config_invalid_yaml(self):
        # Create a temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmpfile:
            tmpfile.write("invalid yaml: -")  # No closing quote
            tmpfile_path = tmpfile.name

        config = load_config(tmpfile_path)
        self.assertIsNone(config)
        os.remove(tmpfile_path)

    def test_load_error_codes_success(self):
        # Create a temporary error codes file
         with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmpfile:
            yaml.dump({
                "errors": {
                    "TestError": {"code": 999, "message": "Test Error Message"}
                }
            }, tmpfile)
            tmpfile_path = tmpfile.name

         error_codes = load_error_codes(tmpfile_path)
         self.assertEqual(error_codes, {999: "Test Error Message"})
         os.remove(tmpfile_path)

    def test_load_error_codes_file_not_found(self):
        with self.assertRaises(ConfigurationError) as context:
            load_error_codes("nonexistent_file.yaml")
        self.assertEqual(context.exception.code, 104)


    def test_load_error_codes_invalid_format(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmpfile:
            yaml.dump({"invalid": "data"}, tmpfile) # Missing errors key
            tmpfile_path = tmpfile.name
        with self.assertRaises(ConfigurationError) as context:
            load_error_codes(tmpfile_path)
        self.assertEqual(context.exception.code, 104)
        os.remove(tmpfile_path)


    def test_load_error_codes_missing_code_or_message(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmpfile:
            yaml.dump({
                "errors": {
                    "TestError": {"message": "Test Error Message"}  # Missing "code"
                }
            }, tmpfile)
            tmpfile_path = tmpfile.name
        with self.assertRaises(ConfigurationError) as context:
            load_error_codes(tmpfile_path)
        self.assertEqual(context.exception.code, 104)
        os.remove(tmpfile_path)


if __name__ == '__main__':
    unittest.main()

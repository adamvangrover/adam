# tests/test_config_utils.py

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml

# Updated import: removed load_error_codes and ConfigurationError
from core.utils.config_utils import load_app_config, load_config


class TestConfigUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_config_")
        # Optional: Disable lower level logging if tests generate too much noise
        # logging.disable(logging.WARNING) 

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        # logging.disable(logging.NOTSET) # Re-enable logging if it was disabled

    def _create_temp_yaml_file(self, filename, data_dict=None, content_str=None):
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w') as f:
            if data_dict is not None:
                yaml.dump(data_dict, f)
            elif content_str is not None:
                f.write(content_str)
            # If both are None, an empty file is created
        return filepath

    # --- Tests for load_config ---
    def test_load_config_valid_yaml(self):
        valid_data = {"key": "value", "number": 123}
        filepath = self._create_temp_yaml_file("valid.yaml", data_dict=valid_data)
        result = load_config(filepath)
        self.assertEqual(result, valid_data)

    def test_load_config_non_existent_file(self):
        # Logging is done to the root logger by default in config_utils
        with self.assertLogs(logger=None, level='ERROR') as cm:
            result = load_config(os.path.join(self.temp_dir, "i_do_not_exist.yaml"))
        self.assertIsNone(result)
        self.assertTrue(any("Config file not found" in log_msg for log_msg in cm.output))

    def test_load_config_empty_yaml(self):
        filepath = self._create_temp_yaml_file("empty.yaml") # Creates an empty file
        with self.assertLogs(logger=None, level='WARNING') as cm:
            result = load_config(filepath)
        self.assertEqual(result, {}) # As per current implementation
        self.assertTrue(any("Configuration file is empty" in log_msg for log_msg in cm.output))
        
    def test_load_config_invalid_yaml(self):
        filepath = self._create_temp_yaml_file("invalid.yaml", content_str="key: value\n  bad_indent: - item1")
        with self.assertLogs(logger=None, level='ERROR') as cm:
            result = load_config(filepath)
        self.assertIsNone(result)
        self.assertTrue(any("Error parsing YAML" in log_msg for log_msg in cm.output))

    # --- Tests for load_app_config ---
    @patch('core.utils.config_utils.load_config')
    def test_load_app_config_basic_merge(self, mock_load_config_func):
        def side_effect_loader(filepath):
            if 'api.yaml' in filepath:
                return {'api': {'host': 'test_host', 'port': 8080}}
            elif 'logging.yaml' in filepath:
                return {'logging': {'level': 'TEST_DEBUG'}}
            return {} 
        
        mock_load_config_func.side_effect = side_effect_loader
        
        app_config = load_app_config()
        
        self.assertIn('api', app_config)
        self.assertEqual(app_config['api']['host'], 'test_host')
        self.assertIn('logging', app_config)
        self.assertEqual(app_config['logging']['level'], 'TEST_DEBUG')
        self.assertGreaterEqual(mock_load_config_func.call_count, 2)


    @patch('core.utils.config_utils.load_config')
    def test_load_app_config_agent_override(self, mock_load_config_func):
        def side_effect_loader(filepath):
            # The order of loading is defined in load_app_config
            # settings.yaml is loaded, then agents.yaml is loaded.
            # The 'agents' key from agents.yaml should overwrite the 'agents' key from settings.yaml
            if 'config/settings.yaml' == filepath:
                return {'settings_specific': 'value', 'agents': {'agent1': {'param_a': 'original_value', 'param_c': 'settings_only'}}}
            elif 'config/agents.yaml' == filepath: 
                return {'agents': {'agent1': {'param_a': 'overridden_value', 'param_b': 'agents_only'}}}
            # Return empty for other files to isolate the test and prevent NoneType errors on .update()
            return {}

        mock_load_config_func.side_effect = side_effect_loader
        
        app_config = load_app_config()
        
        # Check that 'settings_specific' from settings.yaml is present (not overwritten)
        self.assertIn('settings_specific', app_config)
        self.assertEqual(app_config['settings_specific'], 'value')

        # Check the 'agents' configuration
        self.assertIn('agents', app_config)
        self.assertIn('agent1', app_config['agents'])
        agent1_config = app_config['agents']['agent1']
        
        # 'param_a' should be overridden by agents.yaml
        self.assertEqual(agent1_config.get('param_a'), 'overridden_value')
        # 'param_b' should be present from agents.yaml
        self.assertEqual(agent1_config.get('param_b'), 'agents_only')
        # 'param_c' from settings.yaml should NOT be present because the 'agents' dictionary
        # from agents.yaml entirely replaced the 'agents' dictionary from settings.yaml.
        self.assertNotIn('param_c', agent1_config)


    @patch('core.utils.config_utils.load_config')
    def test_load_app_config_file_not_found_continues(self, mock_load_config_func):
        def side_effect_loader(filepath):
            if 'config/api.yaml' == filepath: 
                return None 
            elif 'config/logging.yaml' == filepath:
                return {'logging': {'level': 'INFO_FROM_LOGGING_YAML'}}
            return {}

        mock_load_config_func.side_effect = side_effect_loader
        
        with self.assertLogs(logger=None, level='WARNING') as cm:
            app_config = load_app_config()
            
        self.assertTrue(any("config/api.yaml could not be loaded. Skipping." in log_msg for log_msg in cm.output))
        
        self.assertIn('logging', app_config)
        self.assertEqual(app_config['logging']['level'], 'INFO_FROM_LOGGING_YAML')
        self.assertNotIn('api', app_config)

if __name__ == '__main__':
    unittest.main()

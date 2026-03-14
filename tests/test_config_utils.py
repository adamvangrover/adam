# tests/test_config_utils.py

import os
import yaml
import logging
from pathlib import Path
from unittest.mock import patch
import pytest

from core.utils.config_utils import (
    load_config,
    load_app_config,
    deep_update,
    _substitute_env_vars,
    save_config
)

@pytest.fixture
def temp_yaml_file(tmp_path):
    def _create_yaml(filename, data_dict=None, content_str=None):
        filepath = tmp_path / filename
        if data_dict is not None:
            with open(filepath, 'w') as f:
                yaml.dump(data_dict, f)
        elif content_str is not None:
            filepath.write_text(content_str)
        else:
            filepath.touch()
        return filepath
    return _create_yaml

# --- Tests for _substitute_env_vars ---
def test_substitute_env_vars_with_value():
    os.environ["TEST_VAR"] = "test_value"
    result = _substitute_env_vars("The value is ${TEST_VAR}")
    assert result == "The value is test_value"
    del os.environ["TEST_VAR"]

def test_substitute_env_vars_with_default():
    result = _substitute_env_vars("The value is ${NON_EXISTENT:default_value}")
    assert result == "The value is default_value"

def test_substitute_env_vars_missing_no_default(caplog):
    with caplog.at_level(logging.WARNING):
        result = _substitute_env_vars("The value is ${MISSING_VAR}")
        assert result == "The value is "
        assert "not set and no default provided" in caplog.text

# --- Tests for load_config ---
def test_load_config_valid_yaml(temp_yaml_file):
    valid_data = {"key": "value", "number": 123}
    filepath = temp_yaml_file("valid.yaml", data_dict=valid_data)
    result = load_config(filepath)
    assert result == valid_data

def test_load_config_non_existent_file(caplog):
    with caplog.at_level(logging.ERROR):
        result = load_config("i_do_not_exist.yaml")
        assert result is None
        assert "Config file not found" in caplog.text

def test_load_config_empty_yaml(temp_yaml_file, caplog):
    filepath = temp_yaml_file("empty.yaml")
    with caplog.at_level(logging.WARNING):
        result = load_config(filepath)
        assert result == {}
        assert "Configuration file is empty" in caplog.text

def test_load_config_invalid_yaml(temp_yaml_file, caplog):
    filepath = temp_yaml_file("invalid.yaml", content_str="key: value\n  bad_indent: - item1")
    with caplog.at_level(logging.ERROR):
        result = load_config(filepath)
        assert result is None
        assert "Error parsing YAML" in caplog.text

def test_load_config_not_a_dict(temp_yaml_file, caplog):
    filepath = temp_yaml_file("list.yaml", content_str="- item1\n- item2")
    with caplog.at_level(logging.WARNING):
        result = load_config(filepath)
        assert result == {}
        assert "did not parse as a dictionary" in caplog.text

# --- Tests for deep_update ---
def test_deep_update():
    dict1 = {"a": 1, "b": {"c": 2, "d": 3}}
    dict2 = {"b": {"c": 99, "e": 4}, "f": 5}
    result = deep_update(dict1, dict2)
    assert result == {"a": 1, "b": {"c": 99, "d": 3, "e": 4}, "f": 5}

def test_deep_update_scalar_override_with_dict():
    # If d has scalar and u has dict, the scalar gets overwritten with the dict
    dict1 = {"a": 1}
    dict2 = {"a": {"b": 2}}
    result = deep_update(dict1, dict2)
    assert result == {"a": {"b": 2}}

# --- Tests for load_app_config ---
@patch('core.utils.config_utils.load_config')
def test_load_app_config_basic_merge(mock_load_config):
    def side_effect_loader(filepath):
        if 'api.yaml' in str(filepath):
            return {'api': {'host': 'test_host', 'port': 8080}}
        elif 'logging.yaml' in str(filepath):
            return {'logging': {'level': 'TEST_DEBUG'}}
        return {}
    mock_load_config.side_effect = side_effect_loader
    app_config = load_app_config()
    assert app_config['api']['host'] == 'test_host'
    assert app_config['logging']['level'] == 'TEST_DEBUG'

@patch('core.utils.config_utils.load_config')
def test_load_app_config_agent_override(mock_load_config):
    def side_effect_loader(filepath):
        if 'config/settings.yaml' == str(filepath):
            return {'settings_specific': 'value', 'agents': {'agent1': {'param_a': 'original_value', 'param_c': 'settings_only'}}}
        elif 'config/agents.yaml' == str(filepath):
            return {'agents': {'agent1': {'param_a': 'overridden_value', 'param_b': 'agents_only'}}}
        return {}

    mock_load_config.side_effect = side_effect_loader
    app_config = load_app_config()

    assert app_config['settings_specific'] == 'value'
    agent1_config = app_config['agents']['agent1']
    assert agent1_config.get('param_a') == 'overridden_value'
    assert agent1_config.get('param_b') == 'agents_only'
    # Test should expect param_c to be retained because of deep_update
    assert 'param_c' in agent1_config
    assert agent1_config.get('param_c') == 'settings_only'

@patch('core.utils.config_utils.load_config')
def test_load_app_config_file_not_found_continues(mock_load_config, caplog):
    def side_effect_loader(filepath):
        if 'config/api.yaml' == str(filepath):
            return None
        elif 'config/logging.yaml' == str(filepath):
            return {'logging': {'level': 'INFO_FROM_LOGGING_YAML'}}
        return {}

    mock_load_config.side_effect = side_effect_loader
    with caplog.at_level(logging.WARNING):
        app_config = load_app_config()
        assert "config/api.yaml could not be loaded. Skipping." in caplog.text

    assert app_config['logging']['level'] == 'INFO_FROM_LOGGING_YAML'
    assert 'api' not in app_config

# --- Tests for save_config ---
def test_save_config(tmp_path):
    config_data = {"test": {"nested": "value"}}
    filepath = tmp_path / "new_dir" / "test_save.yaml"
    success = save_config(config_data, filepath)

    assert success is True
    assert filepath.exists()

    with open(filepath, 'r') as f:
        loaded_data = yaml.safe_load(f)
    assert loaded_data == config_data

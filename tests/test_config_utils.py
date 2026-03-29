import os
import pytest
from unittest.mock import patch
from core.utils.config_utils import (
    _substitute_env_vars,
    deep_update,
    load_config,
    load_app_config,
    load_error_codes,
    clear_config_cache,
    save_config
)

def test_substitute_env_vars():
    # Setup mock env vars
    with patch.dict(os.environ, {"API_KEY": "secret123", "PORT": "8080"}):
        # Basic substitution
        assert _substitute_env_vars("Key: ${API_KEY}") == "Key: secret123"
        # Default value fallback
        assert _substitute_env_vars("Timeout: ${TIMEOUT:30}") == "Timeout: 30"
        # Missing without default
        assert _substitute_env_vars("Value: ${MISSING}") == "Value: "
        # Multiple substitutions
        assert _substitute_env_vars("${API_KEY}:${PORT}") == "secret123:8080"

def test_deep_update():
    base_dict = {"a": 1, "b": {"c": 2}}
    update_dict = {"b": {"d": 3}, "e": 4}

    # Simple recursive merge
    result = deep_update(base_dict, update_dict)
    assert result == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    # Overriding dict with scalar or scalar with dict
    base_dict2 = {"a": {"b": 1}}
    update_dict2 = {"a": 2}
    result2 = deep_update(base_dict2, update_dict2)
    assert result2 == {"a": 2}

    base_dict3 = {"a": 1}
    update_dict3 = {"a": {"b": 2}}
    result3 = deep_update(base_dict3, update_dict3)
    assert result3 == {"a": {"b": 2}}

def test_load_config_file_not_found():
    result = load_config("non_existent_file.yaml")
    assert result is None

def test_load_config_empty_file(tmp_path):
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")
    result = load_config(empty_file)
    assert result == {}

def test_load_config_valid(tmp_path):
    yaml_content = "api_key: ${TEST_API_KEY:default_key}\nport: 8080"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    result = load_config(config_file)
    assert result == {"api_key": "default_key", "port": 8080}

@patch('core.utils.config_utils.load_config')
def test_load_app_config(mock_load_config):
    # Mock return values for load_config
    def side_effect(filepath):
        if "api.yaml" in filepath:
            return {"api": {"port": 8080}}
        if "settings.yaml" in filepath:
            return {"app_name": "TestApp"}
        return None

    mock_load_config.side_effect = side_effect

    # Ensure cache is clear before testing
    clear_config_cache()

    config = load_app_config()

    assert config == {"api": {"port": 8080}, "app_name": "TestApp"}

    # Verify caching
    config2 = load_app_config()
    assert config is config2

    # Function is cached, so load_config should not have been called more
    call_count = mock_load_config.call_count

    load_app_config()
    assert mock_load_config.call_count == call_count

@patch('core.utils.config_utils.load_config')
def test_load_error_codes(mock_load_config):
    mock_load_config.return_value = {"errors": {"E001": "Not Found"}}

    clear_config_cache()
    errors = load_error_codes()
    assert errors == {"E001": "Not Found"}

    mock_load_config.return_value = None
    clear_config_cache()
    errors_empty = load_error_codes()
    assert errors_empty == {}

def test_save_config(tmp_path):
    config_data = {"key": "value", "nested": {"a": 1}}
    out_file = tmp_path / "output.yaml"

    success = save_config(config_data, out_file)
    assert success is True
    assert out_file.exists()

    # Verify contents
    content = out_file.read_text()
    assert "key: value" in content
    assert "nested:" in content
    assert "a: 1" in content

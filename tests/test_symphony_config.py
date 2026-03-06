import os
import pytest
from unittest.mock import patch
from core.symphony.config import SymphonyConfig, load_workflow, validate_config, ConfigError

def test_config_defaults():
    config = SymphonyConfig({})
    assert config.tracker_kind == ""
    assert config.tracker_endpoint == ""
    assert config.polling_interval_ms == 30000
    assert config.max_concurrent_agents == 10
    assert config.codex_command == "codex app-server"
    assert config.codex_turn_timeout_ms == 3600000

def test_config_overrides():
    config = SymphonyConfig({
        "tracker": {
            "kind": "linear",
            "api_key": "secret",
            "project_slug": "ENG",
            "active_states": "Todo, Open"
        },
        "agent": {
            "max_concurrent_agents": 5
        }
    })

    assert config.tracker_kind == "linear"
    assert config.tracker_api_key == "secret"
    assert config.tracker_project_slug == "ENG"
    assert config.tracker_active_states == ["Todo", "Open"]
    assert config.tracker_endpoint == "https://api.linear.app/graphql" # default for linear
    assert config.max_concurrent_agents == 5

@patch.dict(os.environ, {"MY_API_KEY": "env-secret"})
def test_config_env_expansion():
    config = SymphonyConfig({
        "tracker": {
            "api_key": "$MY_API_KEY"
        }
    })
    assert config.tracker_api_key == "env-secret"

def test_validate_config():
    valid = SymphonyConfig({
        "tracker": {
            "kind": "linear",
            "api_key": "secret",
            "project_slug": "ENG"
        },
        "codex": {
            "command": "codex app-server"
        }
    })
    assert validate_config(valid) is None

    invalid = SymphonyConfig({
        "tracker": {
            "kind": "linear",
            "api_key": "secret"
        }
    })
    err = validate_config(invalid)
    assert err is not None
    assert err.code == "invalid_config"

def test_load_workflow_parsing(tmp_path):
    workflow_path = tmp_path / "WORKFLOW.md"
    workflow_path.write_text(
        "---\n"
        "tracker:\n"
        "  kind: linear\n"
        "  api_key: secret\n"
        "---\n"
        "You are an agent. Solve {{ issue.identifier }}."
    )

    workflow = load_workflow(str(workflow_path))
    assert workflow.config["tracker"]["kind"] == "linear"
    assert "You are an agent. Solve {{ issue.identifier }}." in workflow.prompt_template

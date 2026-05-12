import pytest
import json
import logging
from core.utils.agent_utils import (
    communicate_between_agents,
    share_knowledge_between_agents,
    monitor_agent_performance,
    validate_agent_inputs,
    format_agent_output,
    log_agent_action,
    parse_json_garbage,
    _MOCK_MESSAGE_BUS
)

@pytest.fixture(autouse=True)
def clear_message_bus():
    """Clear the global message bus before each test."""
    _MOCK_MESSAGE_BUS.clear()
    yield

def test_communicate_between_agents():
    communicate_between_agents("AgentA", "AgentB", "Hello World")
    assert len(_MOCK_MESSAGE_BUS) == 1
    assert _MOCK_MESSAGE_BUS[0] == {
        "sender": "AgentA",
        "receiver": "AgentB",
        "message": "Hello World"
    }

def test_share_knowledge_between_agents():
    share_knowledge_between_agents("AgentX", "AgentY", "sentiment", {"score": 0.8})
    assert len(_MOCK_MESSAGE_BUS) == 1
    assert _MOCK_MESSAGE_BUS[0] == {
        "sender": "AgentX",
        "receiver": "AgentY",
        "knowledge_type": "sentiment",
        "knowledge_data": {"score": 0.8}
    }

def test_monitor_agent_performance(caplog):
    caplog.set_level(logging.INFO)
    monitor_agent_performance("AgentA", "execution_time", 1.23)
    assert "PERFORMANCE | Agent: AgentA | Metric: execution_time | Value: 1.23" in caplog.text

def test_validate_agent_inputs_success():
    inputs = {"param1": "value1", "param2": "value2"}
    validate_agent_inputs("AgentA", inputs, ["param1", "param2"]) # Should not raise

def test_validate_agent_inputs_failure():
    inputs = {"param1": "value1"}
    with pytest.raises(ValueError, match="Agent AgentA missing required parameter: param2"):
        validate_agent_inputs("AgentA", inputs, ["param1", "param2"])

def test_format_agent_output_json():
    data = {"key": "value"}
    result = format_agent_output("AgentA", data, "json")
    assert json.loads(result) == data

def test_format_agent_output_json_unserializable(caplog):
    caplog.set_level(logging.ERROR)
    class Unserializable:
        pass
    result = format_agent_output("AgentA", Unserializable(), "json")
    assert json.loads(result) == {"error": "Unserializable data"}
    assert "JSON serialization failed for agent AgentA:" in caplog.text

def test_format_agent_output_csv_success():
    data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    result = format_agent_output("AgentA", data, "csv")
    assert "name,age" in result
    assert "Alice,30" in result
    assert "Bob,25" in result

def test_format_agent_output_csv_empty():
    result = format_agent_output("AgentA", [], "csv")
    assert result == ""

def test_format_agent_output_csv_invalid_data():
    with pytest.raises(ValueError, match="Agent AgentA output must be a list of dicts for CSV format."):
        format_agent_output("AgentA", {"not": "a list"}, "csv")

def test_format_agent_output_text():
    result = format_agent_output("AgentA", {"key": "value"}, "text")
    assert result == "{'key': 'value'}"

def test_format_agent_output_invalid_format():
    with pytest.raises(ValueError, match="Invalid output format for agent AgentA: invalid"):
        format_agent_output("AgentA", {"key": "value"}, "invalid")

def test_log_agent_action(caplog):
    caplog.set_level(logging.INFO)
    log_agent_action("AgentA", "analyzed_data", "details here")
    assert "ACTION | Agent: AgentA | Action: analyzed_data | Details: details here" in caplog.text

def test_parse_json_garbage_direct():
    text = '{"key": "value"}'
    assert parse_json_garbage(text) == {"key": "value"}

def test_parse_json_garbage_markdown():
    text = '''Here is your data:
```json
{
    "status": "success",
    "score": 100
}
```
Hope this helps!'''
    assert parse_json_garbage(text) == {"status": "success", "score": 100}

def test_parse_json_garbage_greedy():
    text = 'Sure, here is the result: {"status": "success", "nested": {"a": 1}} have a nice day!'
    assert parse_json_garbage(text) == {"status": "success", "nested": {"a": 1}}

def test_parse_json_garbage_failure():
    text = "There is no JSON here."
    with pytest.raises(ValueError, match="Failed to extract valid JSON from the provided text."):
        parse_json_garbage(text)

import pytest
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open, MagicMock

from core.utils.logging_utils import (
    setup_logging, get_logger, get_milestone_logger,
    TraceLogger, SwarmLogger, NarrativeLogger, current_trace_id, JSON_LOGGER_AVAILABLE
)

@pytest.fixture
def temp_log_file(tmp_path):
    log_file = tmp_path / "test_swarm.jsonl"
    yield log_file
    if log_file.exists():
        log_file.unlink()

@pytest.fixture
def reset_swarm_logger():
    """Reset the singleton instance of SwarmLogger between tests."""
    SwarmLogger._instance = None
    yield
    SwarmLogger._instance = None

def test_get_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"

def test_milestone_logger(caplog):
    logger = get_milestone_logger("milestone_test")
    with caplog.at_level(logging.INFO):
        logger.milestone("Test milestone reached")
    assert "✅ Milestone: Test milestone reached" in caplog.text

def test_trace_logger_init():
    trace_id = "test-trace-123"
    logger = TraceLogger(trace_id=trace_id)
    assert logger.trace_id == trace_id
    assert len(logger.get_trace()) == 0

def test_trace_logger_log_step():
    logger = TraceLogger()
    logger.log_step(
        agent_name="AgentA",
        step_name="Step1",
        inputs={"in": 1},
        outputs={"out": 2},
        metadata={"meta": "data"}
    )

    trace = logger.get_trace()
    assert len(trace) == 1
    assert trace[0]["agent"] == "AgentA"
    assert trace[0]["step"] == "Step1"
    assert trace[0]["inputs"] == {"in": 1}
    assert trace[0]["outputs"] == {"out": 2}
    assert trace[0]["metadata"] == {"meta": "data"}
    assert "timestamp" in trace[0]
    assert "trace_id" in trace[0]

def test_trace_logger_clear():
    logger = TraceLogger()
    logger.log_step("A", "S", {}, {})
    assert len(logger.get_trace()) == 1
    logger.clear_trace()
    assert len(logger.get_trace()) == 0

def test_swarm_logger_singleton(temp_log_file, reset_swarm_logger):
    logger1 = SwarmLogger(temp_log_file)
    logger2 = SwarmLogger(temp_log_file)
    assert logger1 is logger2

def test_swarm_logger_log_event(temp_log_file, reset_swarm_logger):
    logger = SwarmLogger(temp_log_file)
    logger.log_event("TEST_EVENT", "AgentB", {"key": "value"})

    assert temp_log_file.exists()
    with temp_log_file.open("r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["event_type"] == "TEST_EVENT"
        assert data["agent_id"] == "AgentB"
        assert data["details"] == {"key": "value"}
        assert "timestamp" in data

def test_swarm_logger_log_thought(temp_log_file, reset_swarm_logger):
    logger = SwarmLogger(temp_log_file)
    logger.log_thought("AgentC", "I am thinking")

    with temp_log_file.open("r") as f:
        data = json.loads(f.readlines()[0])
        assert data["event_type"] == "THOUGHT_TRACE"
        assert data["details"] == {"content": "I am thinking"}

def test_swarm_logger_log_tool(temp_log_file, reset_swarm_logger):
    logger = SwarmLogger(temp_log_file)
    logger.log_tool("AgentD", "calculator", {"eq": "1+1"})

    with temp_log_file.open("r") as f:
        data = json.loads(f.readlines()[0])
        assert data["event_type"] == "TOOL_EXECUTION"
        assert data["details"] == {"tool": "calculator", "parameters": {"eq": "1+1"}}

def test_narrative_logger(caplog):
    logger = NarrativeLogger("NarrativeTest")
    with caplog.at_level(logging.INFO):
        logger.log_narrative(
            event="Market Crash",
            analysis="High volatility",
            decision="Sell",
            outcome="Saved 10%",
            metadata={"source": "Bloomberg"}
        )

    assert "NARRATIVE:" in caplog.text
    assert "Market Crash" in caplog.text
    assert "High volatility" in caplog.text
    assert "Sell" in caplog.text
    assert "Saved 10%" in caplog.text
    assert "Bloomberg" in caplog.text

def test_current_trace_id_context():
    # Set a trace ID in the context
    token = current_trace_id.set("global-trace-xyz")

    try:
        # TraceLogger should pick it up automatically
        t_logger = TraceLogger()
        assert t_logger.trace_id == "global-trace-xyz"

        # NarrativeLogger should use it
        n_logger = NarrativeLogger("NarrativeTest")
        with patch.object(n_logger.logger, 'info') as mock_info:
            n_logger.log_narrative("E", "A", "D", "O")
            # Extract the logged JSON string
            call_args = mock_info.call_args[0][0]
            assert "global-trace-xyz" in call_args

    finally:
        current_trace_id.reset(token)

@patch('core.utils.logging_utils.Path.exists')
@patch('core.utils.logging_utils.yaml.safe_load')
@patch('core.utils.logging_utils.logging.config.dictConfig')
def test_setup_logging_with_file(mock_dict_config, mock_safe_load, mock_exists):
    mock_exists.return_value = True
    mock_safe_load.return_value = {"version": 1}

    # Mock open
    m = mock_open(read_data="version: 1")
    with patch('core.utils.logging_utils.Path.open', m):
        setup_logging(default_path="dummy.yaml")

    mock_dict_config.assert_called_once_with({"version": 1})

def test_setup_logging_with_dict():
    with patch('core.utils.logging_utils.logging.config.dictConfig') as mock_dict_config:
        setup_logging(config={"version": 1})
        mock_dict_config.assert_called_once_with({"version": 1})

@pytest.mark.asyncio
async def test_swarm_logger_async_log_event(temp_log_file, reset_swarm_logger):
    logger = SwarmLogger(temp_log_file)
    await logger.async_log_event("ASYNC_EVENT", "AgentAsync", {"k": "v"})

    assert temp_log_file.exists()
    with temp_log_file.open("r") as f:
        data = json.loads(f.readlines()[0])
        assert data["event_type"] == "ASYNC_EVENT"
        assert data["agent_id"] == "AgentAsync"

@pytest.mark.asyncio
async def test_swarm_logger_async_log_thought(temp_log_file, reset_swarm_logger):
    logger = SwarmLogger(temp_log_file)
    await logger.async_log_thought("AgentC", "Async thinking")

    with temp_log_file.open("r") as f:
        data = json.loads(f.readlines()[0])
        assert data["event_type"] == "THOUGHT_TRACE"
        assert data["details"] == {"content": "Async thinking"}

@pytest.mark.asyncio
async def test_swarm_logger_async_log_tool(temp_log_file, reset_swarm_logger):
    logger = SwarmLogger(temp_log_file)
    await logger.async_log_tool("AgentD", "calculator", {"eq": "2+2"})

    with temp_log_file.open("r") as f:
        data = json.loads(f.readlines()[0])
        assert data["event_type"] == "TOOL_EXECUTION"
        assert data["details"] == {"tool": "calculator", "parameters": {"eq": "2+2"}}

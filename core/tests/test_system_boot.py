import pytest
import os
import json
import logging
from core.system.system_boot_logger import SystemBootLogger, BootLogEntry
from core.system.boot_protocol import BootProtocol
from scripts.boot_system import boot_full_system

TEST_LOG_FILE = "logs/test_version_control_log.jsonl"

@pytest.fixture
def mock_logger():
    # Swap the log file path for testing
    original_log_file = SystemBootLogger.LOG_FILE
    SystemBootLogger.LOG_FILE = TEST_LOG_FILE

    # Clean up before test
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)

    yield

    # Restore after test
    SystemBootLogger.LOG_FILE = original_log_file
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)

def test_boot_logging(mock_logger):
    """Verify that a boot entry is correctly written to the file."""
    entry = BootLogEntry(
        timestamp=1234567890.0,
        agent_id="test_agent_007",
        status="READY",
        highest_conviction_prompt="Execute Order 66",
        conviction_score=1.0
    )

    SystemBootLogger.log_boot(entry)

    assert os.path.exists(TEST_LOG_FILE)

    with open(TEST_LOG_FILE, "r") as f:
        line = f.readline()
        data = json.loads(line)

    assert data["agent_id"] == "test_agent_007"
    assert data["status"] == "READY"
    assert data["highest_conviction_prompt"] == "Execute Order 66"

class MockAgent(BootProtocol):
    def __init__(self, agent_id):
        self.agent_id = agent_id

@pytest.mark.asyncio
async def test_boot_protocol_mixin(mock_logger):
    """Verify the mixin correctly calls the logger."""
    agent = MockAgent("mixin_test_agent")
    agent.report_boot_status(agent.agent_id, "Hello World", 0.5)

    with open(TEST_LOG_FILE, "r") as f:
        line = f.readline()
        data = json.loads(line)

    assert data["agent_id"] == "mixin_test_agent"
    assert data["conviction_score"] == 0.5

@pytest.mark.asyncio
async def test_full_system_boot_script(mock_logger):
    """Verify the boot script runs and logs multiple entries."""
    await boot_full_system()

    with open(TEST_LOG_FILE, "r") as f:
        lines = f.readlines()

    # Expecting at least 3 entries: Sentiment, Fundamental, Worker
    assert len(lines) >= 3

    ids = [json.loads(l)["agent_id"] for l in lines]
    assert "market_sentiment_01" in ids
    assert "fundamental_analyst_01" in ids
    # Worker ID is random uuid, so we just check count

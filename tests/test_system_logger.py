import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.utils.system_logger import SystemLogger, create_timestamped_system_file


@pytest.fixture
def temp_log_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()

def test_system_logger_init(temp_log_file):
    logger = SystemLogger(log_file=str(temp_log_file))
    assert logger.log_file == temp_log_file
    assert temp_log_file.parent.exists()

def test_system_logger_log_event(temp_log_file):
    logger = SystemLogger(log_file=str(temp_log_file))
    details = {"key": "value"}
    logger.log_event("TEST_TAG", details)

    with open(temp_log_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["tag"] == "TEST_TAG"
    assert event["details"] == details
    assert "timestamp" in event

def test_system_logger_read_events(temp_log_file):
    logger = SystemLogger(log_file=str(temp_log_file))
    logger.log_event("TAG1", {"k": 1})
    logger.log_event("TAG2", {"k": 2})

    # Append malformed JSON
    with open(temp_log_file, "a") as f:
        f.write("invalid json\n")

    events = logger._read_events()
    assert len(events) == 2
    assert events[0]["tag"] == "TAG1"
    assert events[1]["tag"] == "TAG2"

@patch('core.utils.system_logger.create_timestamped_system_file')
def test_system_logger_consolidate_logs(mock_create, temp_log_file):
    logger = SystemLogger(log_file=str(temp_log_file))
    logger.log_event("TAG1", {"k": 1})
    logger.consolidate_logs()

    mock_create.assert_called_once()
    payload = mock_create.call_args[0][0]
    assert "system_events" in payload
    assert len(payload["system_events"]) == 1
    assert payload["system_events"][0]["tag"] == "TAG1"

def test_create_timestamped_system_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "test_system_state.json"

        input_data = {
            "market_mayhem_dec_2025.json": {},
            "market_state.json": {},
            "retail_alpha.json": {}
        }

        create_timestamped_system_file(input_data, output_filename=str(output_file))

        assert output_file.exists()

        with open(output_file, "r") as f:
            data = json.load(f)

        assert "system_metadata" in data
        assert "compilation_timestamp" in data["system_metadata"]

        payload = data["data_payload"]
        assert "market_mayhem_current.json" in payload
        assert "v23_knowledge_graph" in payload["market_mayhem_current.json"]
        assert "meta" in payload["market_mayhem_current.json"]["v23_knowledge_graph"]
        assert "generated_at" in payload["market_mayhem_current.json"]["v23_knowledge_graph"]["meta"]

        assert "metadata" in payload["market_state.json"]
        assert "generated_at" in payload["market_state.json"]["metadata"]

        assert "timestamp" in payload["retail_alpha.json"]

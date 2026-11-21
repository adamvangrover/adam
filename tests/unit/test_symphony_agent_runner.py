import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from core.symphony.config import SymphonyConfig
from core.symphony.workspace import WorkspaceManager
from core.symphony.agent_runner import AgentRunner, AgentError
from core.symphony.models import Issue, WorkflowDefinition

@pytest.fixture
def config(tmp_path):
    return SymphonyConfig({
        "workspace": {
            "root": str(tmp_path / "workspaces")
        },
        "codex": {
            "read_timeout_ms": 1000,
            "turn_timeout_ms": 1000
        }
    })

@pytest.fixture
def issue():
    return Issue(id="1", identifier="ENG-1", title="Task", state="Todo")

@pytest.fixture
def workflow():
    return WorkflowDefinition(config={}, prompt_template="Prompt {{ issue.identifier }}")

class MockProcess:
    def __init__(self, responses):
        self.stdin = MagicMock()
        self.stdout = AsyncMock()
        self.stderr = AsyncMock()
        self.pid = 1234
        self.returncode = None

        self._responses = responses
        self._resp_idx = 0

        async def mock_readline():
            if self._resp_idx < len(self._responses):
                res = self._responses[self._resp_idx]
                self._resp_idx += 1
                return res.encode('utf-8')
            return b""

        self.stdout.readline.side_effect = mock_readline

        # Async methods for drain
        async def mock_drain():
            pass
        self.stdin.drain = mock_drain

    def terminate(self):
        pass

@pytest.mark.asyncio
async def test_agent_runner_success(config, issue, workflow):
    manager = WorkspaceManager(config)
    runner = AgentRunner(config, manager)

    responses = [
        # Init response
        '{"method": "initialized", "params": {}}\n',
        # Thread response
        '{"result": {"thread": {"id": "thread_123"}}}\n',
        # Turn 1 notifications
        '{"method": "notification", "params": {}}\n',
        # Turn 1 Usage update
        '{"method": "thread/tokenUsage/updated", "params": {"total_token_usage": {"total_tokens": 150}}}\n',
        # Turn 1 Completion
        '{"method": "turn/completed", "params": {}}\n'
    ]

    mock_proc = MockProcess(responses)

    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc

        events = []
        def on_event(evt):
            events.append(evt)

        def fetch_cb(i):
            # Tell runner issue state is changed to "Done" so it stops the loop
            return Issue(id="1", identifier="ENG-1", title="Task", state="Done")

        result = await runner.run(workflow, issue, 1, on_event, fetch_cb)

        assert result == "success"

        event_types = [e["event"] for e in events]
        assert "session_started" in event_types
        assert "notification" in event_types
        assert "turn_completed" in event_types

@pytest.mark.asyncio
async def test_agent_runner_timeout(config, issue, workflow):
    manager = WorkspaceManager(config)
    runner = AgentRunner(config, manager)

    # We provide no responses so readline hangs and times out
    mock_proc = MockProcess([])

    # Mock wait_for to raise timeout
    async def mock_wait_for(*args, **kwargs):
        raise asyncio.TimeoutError()

    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_proc
        with patch('asyncio.wait_for', side_effect=mock_wait_for):
            with pytest.raises(AgentError) as exc:
                await runner.run(workflow, issue, 1, lambda e: None, lambda i: issue)

            assert exc.value.code == "response_timeout"

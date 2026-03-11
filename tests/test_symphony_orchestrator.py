import pytest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from core.symphony.orchestrator import SymphonyOrchestrator
from core.symphony.models import Issue, WorkflowDefinition
from core.symphony.config import SymphonyConfig

@pytest.fixture
def orchestrator(tmp_path):
    workflow_path = tmp_path / "WORKFLOW.md"
    workflow_path.write_text(
        "---\n"
        "tracker:\n"
        "  kind: linear\n"
        "  api_key: mock\n"
        "  project_slug: ENG\n"
        "  active_states: Todo, In Progress\n"
        "  terminal_states: Done\n"
        "polling:\n"
        "  interval_ms: 100\n"
        "agent:\n"
        "  max_concurrent_agents: 2\n"
        "  max_concurrent_agents_by_state:\n"
        "    todo: 1\n"
        "---\n"
        "prompt"
    )
    orch = SymphonyOrchestrator(str(workflow_path))
    orch.reload_config()
    orch.workflow = WorkflowDefinition(prompt_template="prompt")
    orch.tracker = MagicMock()
    orch.workspace_manager = MagicMock()
    orch.agent_runner = MagicMock()
    orch._loop = asyncio.get_event_loop()
    return orch

@pytest.mark.asyncio
async def test_slot_limits(orchestrator):
    # Todo is limited to 1 per state map
    issue1 = Issue(id="1", identifier="ENG-1", title="1", state="Todo")
    issue2 = Issue(id="2", identifier="ENG-2", title="2", state="Todo")
    issue3 = Issue(id="3", identifier="ENG-3", title="3", state="In Progress")

    assert orchestrator._has_available_slots(issue1) is True

    # Fake running issue 1
    orchestrator.state.running["1"] = MagicMock()
    orchestrator.state.running["1"].issue = issue1

    # Now issue 2 (todo) should fail state limits
    assert orchestrator._has_available_slots(issue2) is False

    # Issue 3 (in progress) should pass
    assert orchestrator._has_available_slots(issue3) is True

@pytest.mark.asyncio
async def test_dispatch(orchestrator):
    issue = Issue(id="1", identifier="ENG-1", title="1", state="Todo")

    # Mock task creation
    with patch('asyncio.create_task') as mock_task:
        await orchestrator._dispatch_issue(issue, attempt=None)

    assert "1" in orchestrator.state.running
    assert "1" in orchestrator.state.claimed
    assert mock_task.called

@pytest.mark.asyncio
async def test_worker_exit_normal(orchestrator):
    issue = Issue(id="1", identifier="ENG-1", title="1", state="Todo")

    # Mock dispatch
    with patch('asyncio.create_task'):
        await orchestrator._dispatch_issue(issue, attempt=None)

    # Exit normal
    await orchestrator._on_worker_exit("1", "normal")

    assert "1" not in orchestrator.state.running
    assert "1" in orchestrator.state.completed
    assert "1" in orchestrator.state.retry_attempts
    assert orchestrator.state.retry_attempts["1"].attempt == 1

@pytest.mark.asyncio
async def test_worker_exit_abnormal(orchestrator):
    issue = Issue(id="1", identifier="ENG-1", title="1", state="Todo")

    with patch('asyncio.create_task'):
        await orchestrator._dispatch_issue(issue, attempt=None)

    await orchestrator._on_worker_exit("1", "abnormal", "some error")

    assert "1" not in orchestrator.state.running
    assert "1" not in orchestrator.state.completed
    assert "1" in orchestrator.state.retry_attempts
    # attempt should be 1 since start was None
    assert orchestrator.state.retry_attempts["1"].attempt == 1
    assert orchestrator.state.retry_attempts["1"].error == "some error"

@pytest.mark.asyncio
async def test_reconciliation_terminal_state(orchestrator):
    issue = Issue(id="1", identifier="ENG-1", title="1", state="Todo")

    with patch('asyncio.create_task'):
        await orchestrator._dispatch_issue(issue, attempt=None)

    # Tracker returns Done (terminal)
    done_issue = Issue(id="1", identifier="ENG-1", title="1", state="Done")
    orchestrator.tracker.fetch_issue_states_by_ids.return_value = [done_issue]

    # Mock to_thread manually to return immediately
    async def mock_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch('asyncio.to_thread', side_effect=mock_to_thread):
        await orchestrator._reconcile_running_issues()

    assert "1" not in orchestrator.state.running
    orchestrator.workspace_manager.cleanup_workspace.assert_called_with("ENG-1")

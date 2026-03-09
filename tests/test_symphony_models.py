import pytest
from datetime import datetime, timezone
from core.symphony.models import Issue, BlockerRef, OrchestratorRuntimeState

def test_issue_model():
    issue = Issue(
        id="abc123",
        identifier="MT-1",
        title="Test Issue",
        state="Todo"
    )
    assert issue.id == "abc123"
    assert issue.identifier == "MT-1"
    assert issue.state == "Todo"
    assert len(issue.labels) == 0
    assert len(issue.blocked_by) == 0

def test_issue_with_blocker():
    issue = Issue(
        id="abc123",
        identifier="MT-2",
        title="Test Issue 2",
        state="In Progress",
        blocked_by=[BlockerRef(id="def456", identifier="MT-1", state="Todo")]
    )
    assert len(issue.blocked_by) == 1
    assert issue.blocked_by[0].id == "def456"

def test_orchestrator_state():
    state = OrchestratorRuntimeState()
    assert state.poll_interval_ms == 30000
    assert state.max_concurrent_agents == 10
    assert len(state.running) == 0
    assert len(state.claimed) == 0
    assert len(state.retry_attempts) == 0

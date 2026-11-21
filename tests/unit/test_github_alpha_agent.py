import pytest
import os
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime, timedelta
from core.agents.specialized.github_alpha_agent import GitHubAlphaAgent, AgentOutput

# Mock subprocess result
class MockSubprocessResult:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

@pytest.fixture
def agent():
    return GitHubAlphaAgent(config={"agent_id": "test_github_agent"})

@pytest.mark.asyncio
async def test_initialization(agent):
    assert agent.name == "test_github_agent"
    assert agent.temp_dir is not None

@pytest.mark.asyncio
async def test_execute_success(agent):
    repo_url = "https://github.com/test/repo.git"

    # Mock subprocess.run
    with patch("subprocess.run") as mock_run:
        # We have multiple calls to subprocess.run:
        # 1. git clone
        # 2. git rev-list (commit count)
        # 3. git log (unique authors)
        # 4. git log -1 (last commit)

        def side_effect(cmd, **kwargs):
            if "clone" in cmd:
                return MockSubprocessResult()
            elif "rev-list" in cmd:
                return MockSubprocessResult(stdout="120\n") # 120 commits
            elif "log" in cmd and "--format=%aN" in cmd: # unique authors
                return MockSubprocessResult(stdout="Alice\nBob\nCharlie\nAlice\n") # 3 unique
            elif "log" in cmd and "-1" in cmd: # last commit
                # Return yesterday
                yesterday = (datetime.now() - timedelta(days=1)).isoformat()
                return MockSubprocessResult(stdout=yesterday)
            return MockSubprocessResult()

        mock_run.side_effect = side_effect

        # Execute
        result = await agent.execute(query=repo_url)

        # Verify
        assert result["confidence"] > 0.0
        assert result["metadata"]["commit_count"] == 120
        assert result["metadata"]["unique_authors"] == 3
        assert result["metadata"]["signal"] in ["ACTIVE", "HIGH_MOMENTUM"]

        # Check source is correct
        assert result["sources"] == [repo_url]

@pytest.mark.asyncio
async def test_execute_stale_repo(agent):
    repo_url = "https://github.com/stale/repo.git"

    with patch("subprocess.run") as mock_run:
        def side_effect(cmd, **kwargs):
            if "clone" in cmd:
                return MockSubprocessResult()
            elif "rev-list" in cmd:
                return MockSubprocessResult(stdout="5\n")
            elif "log" in cmd and "--format=%aN" in cmd:
                return MockSubprocessResult(stdout="Alice\n")
            elif "log" in cmd and "-1" in cmd:
                # 100 days ago
                past = (datetime.now() - timedelta(days=100)).isoformat()
                return MockSubprocessResult(stdout=past)
            return MockSubprocessResult()

        mock_run.side_effect = side_effect

        result = await agent.execute(query=repo_url)

        assert result["metadata"]["signal"] == "DEAD"
        assert result["confidence"] < 0.2

@pytest.mark.asyncio
async def test_execute_failure(agent):
    repo_url = "https://github.com/bad/repo.git"

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("Git clone failed")

        result = await agent.execute(query=repo_url)

        assert result["confidence"] == 0.0
        assert "error" in result["metadata"]
        assert "Git clone failed" in result["answer"]

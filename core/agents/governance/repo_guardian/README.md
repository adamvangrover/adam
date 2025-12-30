# RepoGuardian Agent

The **RepoGuardian Agent** is a specialized governance agent designed to act as an automated gatekeeper for the repository. It reviews incoming Pull Requests (simulated or real) against a strict set of "Enterprise-Grade" standards.

## Architecture

This module is self-contained within `core/agents/governance/repo_guardian/` and follows the `AgentBase` architecture, making it compatible with both v21 (synchronous) and v22 (asynchronous/message-driven) orchestration patterns.

### Components

- **`agent.py`**: Contains the `RepoGuardianAgent` class. It orchestrates the review process by combining deterministic heuristics (static analysis) with semantic analysis (LLM).
- **`schemas.py`**: Defines robust Pydantic V2 data models for `PullRequest`, `FileDiff`, and `ReviewDecision`.
- **`prompts.py`**: Stores the System Persona and Prompt Templates used for LLM interaction.
- **`tools.py`**: Provides wrappers for Git operations and static analysis utilities.

## Usage

### As a Library

```python
from core.agents.governance.repo_guardian.agent import RepoGuardianAgent
from core.agents.governance.repo_guardian.schemas import PullRequest, FileDiff

agent = RepoGuardianAgent(config={"agent_id": "guardian-1"})

pr = PullRequest(
    pr_id="PR-123",
    author="dev-swarm-01",
    title="Fix infinite loop",
    description="Added a break condition.",
    files=[FileDiff(filepath="main.py", change_type="modify", diff_content="...")]
)

decision = await agent.execute(pr=pr)
print(decision.status) # e.g., "approve"
```

### Integration

To integrate with the `AgentOrchestrator` or `MetaOrchestrator`:
1. Add `RepoGuardianAgent` to the `AGENT_REGISTRY`.
2. Route "review" or "governance" tasks to this agent.

## Future Roadmap

- **Graph Integration**: Implement a `GovernanceGraph` in v23 `langgraph` that uses RepoGuardian as a node.
- **Auto-Fixing**: Enhance the agent to apply the `automated_fixes` returned in the `ReviewDecision` directly to the codebase.
- **Security Scanning**: Integrate real SAST tools (e.g., Bandit, Semgrep) into `tools.py`.

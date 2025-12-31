# RepoGuardian Agent

The **RepoGuardian Agent** is a specialized governance agent designed to act as a gatekeeper for code quality, security, and architectural integrity within the 'Adam' repository. It automates the code review process, combining deterministic static analysis with semantic LLM-based reasoning.

## Capabilities

### 1. Security Scanning
The agent aggressively scans for security vulnerabilities:
- **Secrets Detection**: Identifies potential API keys (AWS, Google, Stripe, etc.), private keys, and generic hardcoded secrets.
- **Dangerous Functions**: Flags usage of functions like `eval()`, `exec()`, and `os.system()` which pose security risks.

### 2. Static Analysis (AST-Based)
Uses Python's Abstract Syntax Tree (AST) to enforce coding standards:
- **Type Hinting**: Checks for missing type annotations on function arguments and return values.
- **Documentation**: Verifies the presence of docstrings for modules, classes, and functions.
- **Complexity**: (Future) Can be extended to check cyclomatic complexity.

### 3. Heuristic Checks
- **File Size**: Warns about excessively large file diffs.
- **Pydantic V2**: Enforces the use of Pydantic V2 patterns (e.g., `field_validator` instead of `validator`).

### 4. LLM-Based Semantic Review
Leverages a Large Language Model to provide high-level insights:
- **Intent Verification**: Checks if the code matches the PR description.
- **Logic & Design**: Reviews the architectural fit and logical correctness.
- **Readability**: Assesses variable naming and code structure.

## Architecture

The agent follows a multi-stage execution pipeline:

1.  **Ingestion**: Receives a `PullRequest` object containing file diffs and metadata.
2.  **Heuristic Analysis**:
    - Runs `SecurityScanner` and `StaticAnalyzer` on each file.
    - Generates a list of `ReviewComment`s and `AnalysisResult`s.
3.  **LLM Review**:
    - Constructs a prompt containing PR details and *factual* analysis results.
    - The LLM synthesizes a final `ReviewDecision`.
4.  **Decision Merging**:
    - Heuristic findings are merged into the LLM's decision.
    - **Critical Overrides**: If critical security issues are found, the agent automatically downgrades the decision to `REJECT` or `REQUEST_CHANGES`, regardless of the LLM's opinion.

## Usage

### CLI

You can run the RepoGuardian against the current repository state or a specific PR (simulated) using the CLI:

```bash
python core/agents/governance/repo_guardian/run_guardian.py --pr-id 123 --author "Alice"
```

### Python API

```python
from core.agents.governance.repo_guardian.agent import RepoGuardianAgent
from core.agents.governance.repo_guardian.schemas import PullRequest, FileDiff

agent = RepoGuardianAgent(config={})

pr = PullRequest(
    pr_id="PR-42",
    author="Bob",
    title="Add new feature",
    description="Implements X",
    files=[
        FileDiff(filepath="main.py", change_type="modify", diff_content="...")
    ]
)

decision = await agent.execute(pr=pr)
print(decision.status)
```

## Extending

- **Adding New Secrets**: Update `SecurityScanner.SECRET_PATTERNS` in `tools.py`.
- **New AST Checks**: Modify `StaticAnalyzer.analyze_python_code` in `tools.py`.
- **Policy Changes**: Update `SYSTEM_PROMPT` in `prompts.py`.

## Testing

Run the included test suite to verify functionality:

```bash
python -m unittest core/agents/governance/repo_guardian/tests/test_tools.py
python -m unittest core/agents/governance/repo_guardian/tests/test_agent.py
```

# How-To Guide: Integrating AI Coding Agents (OpenAI Dots, Claude Code, Claude Cowork, Meta Muse)

This guide provides instructions for connecting, configuring, and developing within the **ADAM Financial Operating System (AFOS v30.1)** using modern agentic coding platforms and AI assistants:
- **OpenAI Dots & Codex Assistants**
- **Anthropic Claude Code (CLI)**
- **Anthropic Claude Cowork (Multi-Agent Collaborative Pairing)**
- **Meta Muse & Muse Spark (Closed-Weights Reasoning & Thought Compression)**

---

## 1. Quick Matrix: Configuration Files

| Assistant / Agent | Configuration / Instruction File | Scope & Role |
| :--- | :--- | :--- |
| **OpenAI Dots / Codex** | `.openai/instructions.md` | General agentic coding, MCP tool bindings, and System 1/2 cognitive awareness. |
| **Claude Code (CLI)** | `CLAUDE.md`, `.claude/config.json` | Terminal-driven autonomous coding, surgical refactoring, and test execution. |
| **Claude Cowork** | `CLAUDE.md`, `.claude/config.json` | Collaborative multi-agent pairing, HITL oversight, and synchronous consensus. |
| **Meta Muse / Muse Spark** | `.meta/muse_guidelines.md` | Thought compression, token budgeting, epistemic verification, and PDIL grounding. |
| **Universal / Frontier** | `LLM_README.md`, `llms.txt`, `mcp.json` | Universal context, Model Context Protocol server, and repo-level rules. |

---

## 2. OpenAI Dots Integration

### Setup & Instruction Loading
OpenAI Dots and automated assistant platforms read `.openai/instructions.md` upon initializing a workspace session. The configuration specifies:
1. **System 1 vs. System 2 Division**: System 1 processes asynchronous real-time ingestion (`core/system/v22_async/`), while System 2 handles DAG workflows (`core/engine/`).
2. **Logic as Data**: All financial rules are stored under `assets/` as `jsonLogic`. Models are strictly forbidden from embedding credit thresholds in Python code.
3. **Execution Commands**:
   ```bash
   uv sync --all-extras
   uv run pytest tests/unit/
   uv run ruff check .
   ```

### Tool Access via MCP
OpenAI models integrate with the repository's Model Context Protocol (MCP) server located at `server/server.py`. Core tools include `financial_statement_parser`, `arbitration_calculator`, and `covenant_headroom_calculator`.

---

## 3. Anthropic Claude Code & Claude Cowork

### Claude Code (CLI)
Claude Code interacts with the repo directly from the developer terminal. It parses `CLAUDE.md` and `.claude/config.json`.
- **Targeted Execution**: Run fast unit test suites via `uv run pytest tests/unit/`.
- **Surgical Edits**: Follow the rule of simplicity—write minimal code, clean up unused imports, and never delete `SUSPENDED_AWAITING_INPUT` markers.
- **Hygiene**: Keep Python files under 350 lines; evict complex prompts to `prompt_library/`.

### Claude Cowork
When operating in Claude Cowork (collaborative multi-agent or human-agent pairing):
1. **Explicit Proposals**: Present the execution plan clearly before executing multi-file edits.
2. **Safety Primacy**: Probabilistic System 1 outputs must be arbitrated by deterministic System 2 checks before finalizing underwriting memos or rating distributions.
3. **Traceability**: Decisions should emit W3C PROV-O audit trails to `.jules/history.jsonl` or temporal state logs.

---

## 4. Meta Muse & Muse Spark

### Thought Compression & Epistemic Grounding
Meta Muse utilizes thought compression for efficient token utilization:
- Inferences must classify claims with epistemic tags: `[VERIFIED]`, `[CALCULATED]`, `[MODEL_EST]`, `[PENDING]`.
- All outputs are validated through the **Probabilistic-to-Deterministic Integration Layer (PDIL)** via `check_grounding()`.
- Conviction scores below `0.85` mandate HITL escalation.

---

## 5. Verification & Compliance Checklist

Before submitting changes produced by any AI coding assistant:
1. **Tests**: Ensure relevant unit and integration tests pass (`uv run pytest tests/unit/`).
2. **Linter**: Ensure zero new lint errors (`uv run ruff check <modified_files>`).
3. **Instruction Sync**: Verify instruction files are in sync (`uv run python scripts/build_instructions.py --verify`).
4. **W3C PROV-O**: Confirm that any modified agents or execution paths maintain cryptographic provenance tracking.

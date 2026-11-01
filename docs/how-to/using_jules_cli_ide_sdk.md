# How-To: Using Jules Helper CLI, Web IDE, and Python SDK

This guide explains how to use **Jules** as a self-contained, offline-capable AI software engineering assistant, web IDE, and automation SDK inside the ADAM OS repository.

---

## Overview

Jules is built directly into ADAM OS to ensure full local autonomy and long-term personal usability even without external cloud dependencies.

### Key Capabilities
- **Multi-Backend LLM Support:** Works offline out-of-the-box using the deterministic `MockLLMProvider`, or connects to local models (`Ollama`) and cloud providers (`OpenAI`, `Anthropic`, `LiteLLM`).
- **Full Local Tool Registry:** Safely executes file edits, directory listing, git status checks, pytest runs, bash commands, persistent memory logging, and W3C PROV-O audit logging.
- **Helper CLI:** Command-line runner for prompt execution (`run`), plan generation (`plan`), interactive REPL (`chat`), system diagnostic (`info`), and web server launch (`serve`).
- **Interactive Web IDE:** Browser-based workbench (`/ide` or `jules_ide.html`) featuring file tree explorer, live file editor, agent chat console, plan tracker, and W3C PROV-O audit stream inspector.
- **Python SDK:** Extensible library (`src/jules`) for integrating Jules into custom automation scripts and workflows.

---

## 1. Using the Helper CLI

Run the CLI using `uv run python scripts/jules_cli.py` or `python -m src.jules.cli`.

### Commands

#### Check System Status & Tools
```bash
uv run python scripts/jules_cli.py info
```

#### Generate a Plan
```bash
uv run python scripts/jules_cli.py plan "Refactor credit risk calculation engine"
```

#### Run an Autonomous Task
```bash
uv run python scripts/jules_cli.py run "List repo files and check git status" --provider mock
```

#### Interactive Chat REPL
```bash
uv run python scripts/jules_cli.py chat
```

#### Launch the Web IDE Server
```bash
uv run python scripts/jules_cli.py serve --host 127.0.0.1 --port 8080
```

---

## 2. Using the Web IDE

1. Start the server: `uv run python scripts/jules_cli.py serve`
2. Open your browser at `http://127.0.0.1:8080` (or double-click `jules_ide.html`).
3. Features:
   - **Agent Workbench:** Formulate execution plans, chat with Jules, and run autonomous tasks.
   - **Code Editor:** Browse workspace files and edit/save code directly.
   - **Audit Stream:** View real-time W3C PROV-O tool execution logs with `context_provenance`.

---

## 3. Using the Python SDK

You can import and use Jules directly in Python:

```python
from src.jules import JulesConfig, JulesAgent, JulesEngine

# Initialize agent with default settings
agent = JulesEngine.create(provider="mock")

# Run interactive chat
reply = agent.chat("What tools are registered?")
print(reply)

# Run an autonomous task loop
result = agent.run("List workspace files and check git status")
print("Status:", result["status"])
print("Plan:", result["plan"])
print("Audit Log Entries:", len(result["audit_log"]))
```

---

## 4. W3C PROV-O Audit Logging

All tool executions invoked by Jules automatically record a W3C PROV-O lineage entry containing `context_provenance` formatted as stringified JSON:

```json
{
  "entity_id": "urn:jules:tool_exec:list_files:2026-10-30T00:00:00Z",
  "tool_name": "list_files",
  "args": {"path": "."},
  "result_status": "success",
  "timestamp": "2026-10-30T00:00:00Z",
  "context_provenance": "{\"prov_type\": \"ToolExecution\", \"tool\": \"list_files\", \"executed_at\": \"2026-10-30T00:00:00Z\", \"agent\": \"JulesAgent\"}"
}
```

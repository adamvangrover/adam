# Jules Tools Reference (`@google/jules`) — AFOS v30.1 Integration Guide

**Jules Tools** is a lightweight command-line interface (CLI) for interacting with **Jules**, Google's autonomous AI coding agent. Within the **ADAM Financial Operating System (AFOS v30.1)** ecosystem, Jules operates as the **Architect Agent (`architect_jules`)** responsible for repository architecture, systemic code refactoring, daily protocol workflow runs, and deterministic execution optimization.

It allows you to manage coding sessions, inspect progress, and integrate Jules into your existing development workflows and scripts directly from your terminal—acting as both a command surface and an interactive dashboard without needing to switch to a web browser.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Authentication](#authentication)
  - [Login](#login)
  - [Logout](#logout)
- [Usage](#usage)
  - [Global Flags](#global-flags)
  - [Available Commands](#available-commands)
- [Interactive Dashboard (TUI)](#interactive-dashboard-tui)
- [AFOS v30.1 Architect Agent (`architect_jules`) Workflows](#afos-v301-architect-agent-architect_jules-workflows)

---

## Installation

To get started, install the tool globally using `npm` or `pnpm`:

```bash
npm install -g @google/jules
```

Once installed, the `jules` command will be available in your terminal.

---

## Authentication

Before you can use the tool, you must authenticate with your Google account.

### Login

```bash
jules login
```

This command will open a browser window to guide you through the Google authentication process.

### Logout

To log out from your account:

```bash
jules logout
```

---

## Usage

The CLI is built around commands and subcommands. You can get help for any command by using the `-h` or `--help` flag.

```bash
# Get general help
jules help

# Get help for a specific command (e.g., remote)
jules remote --help
```

### Global Flags

- `-h`, `--help`: Displays help information for `jules` or a specific command.
- `--theme <string>`: Sets the theme for the terminal user interface (TUI). Options are `dark` (default) or `light`.

**Example:**

```bash
jules --theme light
```

---

### Available Commands

#### `version`

Shows the currently installed version of the Jules Tools CLI.

```bash
jules version
```

#### `remote`

The `remote` command is the primary way to interact with Jules sessions running in the cloud. It has several subcommands:

##### `remote list`

Lists your connected repositories or active sessions.

- `--repo`: Flag to list all repositories connected to Jules.
- `--session`: Flag to list all your remote sessions.

**Examples:**

```bash
# List all connected repositories
jules remote list --repo

# List all active and past sessions
jules remote list --session
```

##### `remote new`

Creates a new remote session to delegate a task to Jules.

Jules can automatically infer the repository from your current working directory, so you can often omit the `--repo` flag.

- `--repo <repo_name>`: Specifies the repository for the session (e.g., `adamvangrover/adam`, `torvalds/linux`, or `.` for the current directory's repo).
- `--session "<prompt>"`: A string describing the task for Jules to perform.
- `--parallel <number>`: Starts multiple parallel sessions to work on the same task.

**Examples:**

```bash
# Start a new session to write unit tests in the 'torvalds/linux' repo
jules remote new --repo torvalds/linux --session "write unit tests"

# Delegate an AFOS v30.1 P0 security hardening task in the current repository
jules remote new --repo adamvangrover/adam --session "Audit importlib usage and enforce core.security.safe_unpickler across core/"

# Launch 3 parallel sessions for System 2 DAG optimization
jules remote new --repo . --session "Profile and optimize System 2 DAG planner execution traces with W3C PROV-O headers" --parallel 3
```

##### `remote pull`

Pulls the results (e.g., code changes) from a completed session.

- `--session <session_id>`: The ID of the session you want to pull.

**Example:**

```bash
# Pull the results for session ID 123456
jules remote pull --session 123456
```

#### `completion`

Generates an autocompletion script for your shell (e.g., `bash`, `zsh`) to enable tab completion for `jules` commands.

```bash
# Generate completion script for bash
jules completion bash

# Generate completion script for zsh
jules completion zsh
```

---

## Interactive Dashboard (TUI)

For a more interactive, visual experience, you can launch the Terminal User Interface (TUI) by running the `jules` command without any arguments:

```bash
jules
```

The TUI provides a dashboard view of your sessions, a side-by-side diff viewer for reviewing changes, and guided flows for creating new ones, similar to the web UI.

---

## AFOS v30.1 Architect Agent (`architect_jules`) Workflows

Within ADAM v30.1, Jules is codified as the **Architect Agent (`architect_jules`)** in [`core/agents/architect_agent.py`](../core/agents/architect_agent.py) and orchestrated via [`scripts/run_jules_workflows.py`](../scripts/run_jules_workflows.py).

### Architectural Guardrails for Jules Sessions

1. **Path A vs. Path B Bifurcation:**
   - **Path A (`core/`, `adam_*/`, `src/`):** Strict Pydantic typing, defensive error handling, and W3C PROV-O telemetry (`ProvenanceHeader`).
   - **Path B (`experimental/`, `research/`, `tinker_lab/`):** Rapid prototyping; never imported into Path A modules.
2. **Front-End Isolation:**
   - Never import UI libraries (`streamlit`, `react`) inside `core/` or `adam_*/` execution modules. Presentation stays in `frontend/`, `services/webapp/`, or `showcase/`.
3. **Deterministic Math Delegation:**
   - Never perform domain arithmetic in prompts—defer math to deterministic Rust/Python pricing engines and `jsonLogic` rules.

### Running Automated AFOS Jules Workflows

You can preview or dispatch predefined backlog workflows from [`docs/AGENTS_KNOWLEDGE_BASE.md`](AGENTS_KNOWLEDGE_BASE.md) using the AFOS CLI bridge:

```bash
# List predefined AFOS architectural workflows for Jules
uv run python scripts/run_jules_workflows.py --list-presets

# Dry-run a Jules session with W3C PROV-O audit bundle output
uv run python scripts/run_jules_workflows.py --preset p0_security_hardening --parallel 2

# Dispatch a custom Jules session (dry-run by default; pass --execute to invoke CLI)
uv run python scripts/run_jules_workflows.py --repo adamvangrover/adam --session "Write unit tests for PDIL gatekeeper" --parallel 1
```


# ADAM v23.0 UI User Guide

The ADAM v23.0 system features a completely overhauled user interface designed to provide real-time synthesis, analysis, and representation of the repository's status and architecture.

## Overview

The UI is split into two modes:
1.  **Static Mode:** Works directly from the file system. Displays the repository structure and static agent definitions.
2.  **Live Mode:** Requires the UI Backend Server. Enables real-time system stats, log viewing, and file content inspection.

## Quick Start

To launch the full experience (Live Mode):

```bash
./run_ui.sh
```

This will:
1.  Generate the latest system snapshot (`ui_data.json`).
2.  Start the Flask backend server on `http://localhost:5000`.

## Components

### Mission Control (`index.html`)
The central hub showing system health (CPU/Memory), active agents, and architectural components (v21/v22/v23).

### Navigator (`navigator.html`)
A robust file explorer that allows you to browse the entire repository.
*   **Static Mode:** View file tree only.
*   **Live Mode:** View full file contents with syntax highlighting.

### Agent Matrix (`agents.html`)
A visual grid of all registered agents (Sub-Agents, Meta-Agents, Orchestrators) derived from `AGENTS.md`. It displays their operational status and descriptions.

### Knowledge Graph (`graph.html`)
An interactive neural topology visualization showing the relationships between the Core, Orchestrators, and Sub-Agents.

## Architecture

The UI is built with:
*   **Frontend:** HTML5, Tailwind CSS (via CDN), Vanilla JavaScript.
*   **Backend:** Python/Flask (`services/ui_backend.py`).
*   **Data Layer:** JSON-based state (`scripts/generate_ui_data.py`).

## Troubleshooting

*   **"Connection Offline":** Ensure `run_ui.sh` is running. If you want to use Static Mode, this is normal.
*   **"File Content Not Available":** You are in Static Mode. Run the backend server to enable file reading.

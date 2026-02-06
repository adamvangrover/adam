# Adam v26.0 Setup Guide

This document provides definitive instructions for setting up the Adam environment. We strictly support **`uv`** for dependency management.

---

## 📋 Table of Contents
*   [Prerequisites](#prerequisites)
*   [Environment Variables](#environment-variables)
*   [Installation (Local)](#installation-local)
*   [Docker Deployment](#docker-deployment)
*   [Troubleshooting](#troubleshooting)

---

## Prerequisites

*   **Operating System:**
    *   **Linux:** Ubuntu 22.04 LTS (Recommended)
    *   **macOS:** Ventura or newer (Apple Silicon supported)
    *   **Windows:** WSL2 (Ubuntu 22.04) ONLY. Native Windows is not actively supported.
*   **Python:** 3.10+
*   **Tools:** `uv`, `git`, `curl`, `make` (optional)

---

## Environment Variables

Adam requires a `.env` file in the project root to store secrets.

1.  **Copy the Template:**
    ```bash
    cp .env.example .env
    ```

2.  **Configure Keys:**
    Open `.env` and set the following:

    ```ini
    # --- LLM Providers ---
    OPENAI_API_KEY=sk-...         # Required for Planner/Reasoning
    ANTHROPIC_API_KEY=sk-ant...   # Optional (Recommended for Coding)

    # --- Financial Data ---
    FMP_API_KEY=...               # Financial Modeling Prep (Market Data)
    SEC_API_KEY=...               # Optional (for 10-K fetching)

    # --- Infrastructure ---
    POSTGRES_URI=postgresql://user:pass@localhost:5432/adam
    REDIS_URL=redis://localhost:6379/0
    ```

---

## Installation (Local)

### 1. Install `uv`
If you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*Restart your shell after installation.*

### 2. Clone the Repository
```bash
git clone https://github.com/adamvangrover/adam.git
cd adam
```

### 3. Sync Dependencies
This command creates the virtual environment (`.venv`) and installs all locked dependencies.
```bash
uv sync
```

### 4. Activate Environment
```bash
source .venv/bin/activate
```

### 5. Verify Installation
Run the help command to ensure the CLI is working.
```bash
python scripts/run_adam.py --help
```

---

## Docker Deployment

For production or isolated testing, use Docker Compose.

```bash
# 1. Build and Start
docker-compose up --build -d

# 2. View Logs
docker-compose logs -f app

# 3. Stop
docker-compose down
```

The Web Interface will be available at `http://localhost:80` (or `http://localhost:3000` depending on configuration).

---

## Troubleshooting

### 1. `uv: command not found`
*   **Cause:** The installer didn't update your PATH.
*   **Fix:** Add `export PATH="$HOME/.cargo/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`.

### 2. `libmagic` Error (ImportError: failed to find libmagic)
*   **Cause:** The `python-magic` library requires a system-level dependency for file type detection.
*   **Fix:**
    *   **Ubuntu:** `sudo apt-get install libmagic1`
    *   **macOS:** `brew install libmagic`

### 3. Docker "Port Already in Use"
*   **Cause:** Another service is running on port 5000 or 5432.
*   **Fix:**
    1.  Find the process: `sudo lsof -i :5000`
    2.  Kill it: `kill -9 <PID>`
    3.  Or, change the port mapping in `docker-compose.yml`.

### 4. LLM "Rate Limit Exceeded"
*   **Cause:** You ran out of OpenAI credits.
*   **Fix:** Check your billing status or switch to a local model in `config/models.yaml`.

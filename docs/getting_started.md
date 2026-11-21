# Getting Started with ADAM

This guide walks through setting up ADAM and running a first analysis. For every other setup
(orchestrator only, credit_sentinel only, Rust, frontend, Docker, bundles) see the
[Setup Guide](setup_guide.md).

## Prerequisites

*   **Python 3.12** (3.11 supported; `pyproject.toml` requires `>=3.11,<3.13`).
*   **uv**: We use `uv` for fast, reproducible dependency management.
    *   Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`)
*   **Docker** (Optional but recommended for full stack deployment).
*   **API Keys** (optional): without a key the orchestrator uses mock behaviour. The default model is `gemini-4-neon` (`GEMINI_API_KEY`).

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adamvangrover/adam.git
cd adam
```

### 2. Environment Setup with `uv`

Adam uses a `pyproject.toml` and `uv.lock` to manage dependencies strictly.

```bash
# Create a Python 3.12 environment and sync the orchestrator extras
uv venv --python 3.12
uv sync --extra data-science --extra orchestration
```

This creates `.venv/`. `requirements/base.txt` has conflicting pins — if an import still fails, install
that package individually with `uv pip install <package>`.

### 3. Configure Environment Variables

Copy the example environment file and add your keys.

```bash
cp .env.example .env
```

Open `.env` and set the keys you have (all optional):

```properties
GEMINI_API_KEY=your-key-here
OPENAI_API_KEY=sk-your-key-here
# Optional: Set other keys if needed (e.g., SERPER_API_KEY for search)
```

### 4. VS Code Configuration (Recommended)

To get the best development experience with type checking and auto-completion:

1.  **Install Extensions:**
    *   Python (ms-python.python)
    *   Pylance (ms-python.vscode-pylance)
    *   Ruff (charliermarsh.ruff) - *Fast linting*

2.  **Configure `.vscode/settings.json`:**
    Create or update this file to point VS Code to the `uv` virtual environment.

    ```json
    {
      "python.defaultInterpreterPath": ".venv/bin/python",
      "python.analysis.typeCheckingMode": "strict",
      "python.analysis.extraPaths": ["core", "services"]
    }
    ```

### 5. Sanity Check

Activate the environment and run a quick test to ensure everything is connected.

```bash
source .venv/bin/activate
# Windows: .venv\Scripts\activate

.venv/bin/python tests/evals/run_accuracy.py          # routing + entity accuracy, no API key needed
uv run python scripts/run_adam.py --query "Hello, Adam."
```

**Expected Output:**
You should see logs initializing the `MetaOrchestrator` and a response similar to:
`[Adam]: System is online. How can I assist you with your financial analysis today?`

## Running the System

### Option A: The Neural Dashboard (Recommended)

To see Adam in action, we recommend launching the showcase dashboard.

1.  **Launch the Frontend:**
    Open `showcase/index.html` in your browser. This provides a visual interface to the pre-generated data and simulation capabilities.

2.  **Run the Live Backend (Optional):**
    For live interaction, start the Flask API (port 5001) or the Streamlit valuation app:
    ```bash
    uv sync --extra async-web && uv run python services/webapp/api.py
    uv run streamlit run app.py
    ```

### Option B: Command Line Interface (CLI)

You can interact with agents directly via the CLI for quick queries or debugging.

```bash
# Interactive Mode
uv run python scripts/run_adam.py

# Single Shot
uv run python scripts/run_adam.py --query "Analyze the credit risk of Tesla"
```

## Next Steps

*   Check out the [Tutorials](tutorials.md) to learn how to run specific analyses.
*   Read the [Setup Guide](setup_guide.md) for advanced configuration.

## Interaction modes

*Adapted from the v23 user guide (`docs/archive/v23_manual/user_guide.md`); the route names below are
the current ones in `core/engine/meta_orchestrator.py`.*

You don't pick a mode — `MetaOrchestrator` matches your query against ordered word-boundary patterns:

| Intent words (examples) | Route | Example query |
|---|---|---|
| "deep dive", "full analysis", "valuation", "covenants" | `DEEP_DIVE` | "Deep dive on AAPL valuation" |
| "swarm", "hive", "scouts" | `SWARM` | "Send the swarm over TMT names" |
| "attack", "adversarial", "red team", "wargame" | `RED_TEAM` | "Red team our payment gateway" |
| "crisis", "shock", "stress test", "black swan", "market crash" | `CRISIS` | "Stress test a 60/40 portfolio against a rate shock" |
| "surveillance", "zombie", "distressed" | `SURVEILLANCE` | "Surveillance on distressed names in the book" |
| "ESG", "sustainability", "carbon" | `ESG` | "What is the ESG profile of XOM?" |
| "compliance", "KYC", "AML", "Basel" | `COMPLIANCE` | "Check JPM for Basel III compliance" |
| "analyze", "strategy" | `HIGH` (general planning) | "Analyze the credit risk of TSLA" |

Family-office routes (`FO_WEALTH`, `FO_DEAL`, `FO_EXECUTION`, `FO_MARKET`), `CODE_GEN` and
`MEDIUM`/`LOW` cover the rest; `CODE_GEN` and `SWARM` are matched first.

Queries with no resolvable ticker or entity return `{"status": "clarification_needed"}`. The exact
patterns and order are in `ROUTE_PATTERNS`; `tests/evals/datasets/` holds the labelled examples.

**Reading results.** Outputs are `AgentOutput` objects: `answer`, `sources`, `confidence` and a
`provenance_trace` ([PROVENANCE.md](PROVENANCE.md)). A self-critique/revise step (reflector graph) is
normal. Confidence ≥ 0.85 is autonomous, 0.50–0.8499 waits for human approval, < 0.50 aborts.


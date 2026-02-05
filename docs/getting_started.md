# Getting Started with Adam v26.0

This guide will walk you through setting up the Adam environment and running your first analysis.

## Prerequisites

*   **Python 3.10+**: Ensure you have a compatible Python version installed.
*   **uv**: We use `uv` for fast, reproducible dependency management.
    *   Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`)
*   **Docker** (Optional but recommended for full stack deployment).
*   **API Keys**: You will need an OpenAI API key for the core reasoning engine.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/adamvangrover/adam.git
cd adam
```

### 2. Environment Setup with `uv`

Adam uses a `pyproject.toml` and `uv.lock` to manage dependencies strictly.

```bash
# Create and sync the virtual environment
uv sync
```

This command will create a `.venv` directory and install all required packages (including dev dependencies).

### 3. Configure Environment Variables

Copy the example environment file and add your keys.

```bash
cp .env.example .env
```

Open `.env` and set your `OPENAI_API_KEY`:

```properties
OPENAI_API_KEY=sk-your-key-here
# Optional: Set other keys if needed (e.g., SERPER_API_KEY for search)
```

### 4. Verify Installation

Activate the environment and run a quick test.

```bash
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python scripts/run_adam.py --query "Hello, Adam."
```

You should see a response indicating the system is operational.

## Running the System

### Option A: The Neural Dashboard (Recommended)

To see Adam in action, we recommend launching the showcase dashboard.

1.  **Launch the Frontend:**
    Open `showcase/index.html` in your browser. This provides a visual interface to the pre-generated data and simulation capabilities.

2.  **Run the Live Backend (Optional):**
    For live interaction, you can start the Flask server:
    ```bash
    python app.py
    ```

### Option B: Command Line Interface (CLI)

You can interact with agents directly via the CLI for quick queries or debugging.

```bash
# Interactive Mode
python scripts/run_adam.py

# Single Shot
python scripts/run_adam.py --query "Analyze the credit risk of Tesla"
```

## Next Steps

*   Check out the [Tutorials](tutorials.md) to learn how to run specific analyses.
*   Read the [Setup Guide](setup_guide.md) for advanced configuration.

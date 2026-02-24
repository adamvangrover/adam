# Alternative Setup & Installation Guides (Legacy)

> **⚠️ NOTE:** This document covers alternative and legacy methods for setting up older versions of Adam (e.g., v23.5).
> For the **recommended modern setup** (v26.0+) using `uv`, please refer to the [**Official Setup Guide**](setup_guide.md).

This document covers alternative methods for setting up Adam, including Docker, legacy scripts, and manual pip installation.

## 1. Docker Deployment ("Mission Control")

This is the preferred method for full system demonstrations and UI validation, as it isolates the environment and ensures all dependencies (including Redis) are correctly configured.

### Prerequisites
* Docker & Docker Compose
* NVIDIA Container Toolkit (optional, for GPU support)

### Steps
1. **Build the Container**
   ```bash
   # Build the optimized modern container
   docker build -f Dockerfile.modern -t adam-v23 .
   ```

2. **Run the Container**
   ```bash
   # Run with port forwarding
   docker run -p 3000:3000 -p 8000:8000 \
     -e OPENAI_API_KEY=sk-... \
     adam-v23
   ```
   
   If you have a GPU available:
   ```bash
   docker run --gpus all -p 3000:3000 -p 8000:8000 \
     -e OPENAI_API_KEY=sk-... \
     adam-v23
   ```

## 2. Interactive Setup Wizard

The repository includes an interactive Python script that guides you through dependency checking, API key configuration, and system launch.

```bash
python3 scripts/setup_interactive.py
```

## 3. Legacy Shell Scripts

For users on Linux/macOS who prefer standard shell scripts:

```bash
# Full setup and run
./run_adam.sh
```

## 4. Manual Installation (Pip)

If you prefer to manage your environment manually with standard `pip`:

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install the package in editable mode
pip install -e .

# 3. Run the engine
python core/main.py
```

## 5. Financial Engineering Platform (Streamlit)

To run the standalone Financial Engineering Engine (Valuation, Credit Risk) without the full agentic swarm:

```bash
# Ensure you have streamlit installed
pip install streamlit

# Run the app
streamlit run app.py
```

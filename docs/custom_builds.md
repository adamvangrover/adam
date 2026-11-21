# ADAM v26.0 :: Custom Build System

> Consolidated overview: [setup_guide.md](setup_guide.md). This page keeps the detailed custom builds notes; where they differ, `setup_guide.md` reflects the current tree.

The **Adam Custom Build System** allows you to create tailored, portable environments of the platform. Instead of deploying the entire monolithic repository, you can select specific modules (e.g., just the 13F Tracker or the Simulation Dashboard) and generate a self-contained package with its own dependencies and Docker configuration.

## 🚀 Quick Start

Run the interactive builder wizard:

```bash
python3 scripts/build_adam.py
```

Follow the on-screen prompts to:
1.  **Select Modules:** Choose which components to include (e.g., `market_mayhem`, `repository`).
2.  **Select Profile:** Choose the runtime environment:
    *   **Lite:** HTML/JS only. Best for static hosting (GitHub Pages, S3).
    *   **Standard:** Python + Flask. Best for local development.
    *   **Full:** Docker + ML Stack. Best for production deployment.

## 📦 Output Structure

Builds are generated in the `builds/` directory with a timestamped folder name (e.g., `builds/adam_build_20250501_120000`).

```text
builds/adam_build_TIMESTAMP/
├── index.html              # Landing page linking to all modules
├── requirements.txt        # Generated based on selected profile
├── Dockerfile              # Generated based on selected profile
├── run_module.py           # Universal launcher
├── market_mayhem/          # Exported Module 1
├── repository/             # Exported Module 2
└── ...
```

## 🛠️ Profiles Explained

| Profile | Description | Use Case |
| :--- | :--- | :--- |
| **Lite** | minimal static files. No Python backend required for viewing. | Static hosting, sharing reports via email/zip. |
| **Standard** | Includes `flask`, `requests`, and basic utils. | Local analysis, running the dashboard on a laptop. |
| **Full** | Includes `torch`, `pandas`, `scikit-learn`, `web3`. | heavy lifting, running new simulations, ML inference. |

## 🐳 Docker Support

If you select the **Full** profile, a `Dockerfile` is automatically generated. You can build and run it immediately:

```bash
cd builds/adam_build_TIMESTAMP
docker build -t my-adam-build .
docker run -p 8000:8000 my-adam-build
```

## 🔧 Extending the Builder

The builder logic is located in `scripts/build_adam.py`.
It utilizes `scripts/export_module.py` for the core asset copying logic.
Templates for `requirements.txt` and `Dockerfile` are located in `scripts/templates/`.

To add a new module, update the `MODULES` dictionary in `scripts/export_module.py`.
To update the default dependencies, edit `scripts/templates/requirements_*.txt`.

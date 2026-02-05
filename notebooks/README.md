# Adam Research Notebooks

The `notebooks/` directory serves as the "Laboratory" for Data Scientists and Quants.

## 🧪 Usage Policy
*   **Experimental:** Code here is for research and prototyping. It is **not** production-ready.
*   **Isolated:** Notebooks should not depend on local relative paths if possible, or should explicitly set `PYTHONPATH`.

## 📂 Categories

### 1. Research (`research/`)
Experiments with new models or algorithms.
*   `simulation.ipynb`: Prototyping the One-Shot World Model logic.
*   `walkthrough.ipynb`: A step-by-step guide to the internal reasoning graph.

### 2. Demos (`demos/`)
Polished examples for stakeholders.
*   `fdt_bundle.ipynb`: Demonstrates the Financial Digital Twin capabilities.

## 🚀 Running Notebooks
Ensure your virtual environment is active and `jupyter` is installed.

```bash
source .venv/bin/activate
pip install jupyter
jupyter notebook
```

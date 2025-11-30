# Adam v23.0 Showcase Guide

This guide explains how to demonstrate the key capabilities of the Adam v23.0 Adaptive System.

## 1. The Neural Dashboard
**File:** `showcase/index.html`

The Dashboard is the "Face" of Adam. It simulates a live connection to the Neural Core.

### Key Features to Highlight:
- **Live System Activity:** Watch the "Activity Log" populate with events like "Risk Analysis" and "Red Team Simulation".
- **Component Status:** Shows the health of the `Meta Orchestrator` and `Unified KG`.
- **Financial Terminal:** Links to `dashboard.html` for deep-dive financial charts.

## 2. The Cyclical Reasoning Engine
**File:** `core/v23_graph_engine/cyclical_reasoning_graph.py`

This is the "Brain". It demonstrates **System 2 Thinking** (Slow, Deliberate).

### How it works:
1.  **Retrieve:** Fetches mock financial data (e.g., for Tesla or Apple).
2.  **Draft:** Generates an initial risk assessment.
3.  **Critique:** A separate "Critic" node reviews the draft for missing sections (e.g., "Liquidity Risk").
4.  **Refine:** The system fixes the errors automatically.
5.  **Loop:** This continues until the Quality Score > 0.90.

### Running the Verification:
Run the verification script to see the loop in action via CLI logs:
```bash
python tests/verify_v23_full.py
```
Look for logs indicating `--- Node: Critique ---` and `--- Node: Correction ---`.

## 3. Neuro-Symbolic Planning
**File:** `core/v23_graph_engine/neuro_symbolic_planner.py`

Demonstrates how Adam "thinks" before acting.

- **Symbolic:** Uses the Knowledge Graph to find valid paths (e.g., Company -> Industry -> Risk).
- **Neural:** Uses LLMs to navigate unstructured queries.

## 4. Red Team Simulation
**Concept:**
Adam creates "What-If" scenarios to test resilience. In the showcase, look for "Red Team" events in the dashboard, simulating cyber attacks or macro shocks.

---

## Troubleshooting
- **Missing Data?** The UI uses `showcase/js/mock_data.js` if the backend is offline. Ensure this file exists.
- **Python Errors?** Ensure all dependencies in `requirements.txt` are installed. Use `scripts/run_adam.py` to verify the environment.

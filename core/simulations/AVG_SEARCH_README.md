# AVG Search (AdamVanGrover)

**AVG Search** is a hybrid Quantum-Classical search engine designed for enterprise-scale unstructured data retrieval. It synthesizes Quantum Annealing (QA) with AI-driven optimization (Adam) to probabilistically filter petabyte-scale datasets.

## Core Concepts

1.  **AVG Framework:** Uses the Adam optimizer to tune the quantum annealing schedule ($A(t)$, $B(t)$) to approximate the Roland-Cerf local adiabatic schedule. This allows the system to slow down near the minimum spectral gap, boosting the success probability of finding the ground state (the "needle").
2.  **Hybrid Indexing:** Instead of a single deterministic result, the quantum processor returns a "batch" of high-energy candidates (Collapsed States).
3.  **Classical Verification:** A classical O(1) oracle verifies the batch to identify the true target, discarding noise and local minima.

## Architecture

### 1. Simulation Engine (`core/simulations/avg_search.py`)
*   **`AVGSearch`**: The physics engine that simulates the time-dependent Hamiltonian evolution.
*   **`AdamOptimizer`**: A custom implementation of the Adam algorithm used to iteratively update the schedule parameters based on simulation fidelity (loss).

### 2. Specialized Agent (`core/agents/specialized/quantum_search_agent.py`)
*   **`QuantumSearchAgent`**: Interfaces with the simulation engine.
*   **Skills**:
    *   `run_quantum_search`: Runs the optimization loop.
    *   `run_hybrid_search`: Executes the full pipeline (Optimization -> Sampling -> Verification).

### 3. Visualization (`showcase/quantum_search.html`)
*   Provides a dashboard to visualize the annealing schedules, optimization loss, and the hybrid pipeline flow.
*   Displays the "Enterprise Odds" improvement over classical baseline ($10^{-15} \to 10^{-6}$).

## Usage

### Running the Simulation
To generate new simulation data for the dashboard:
```bash
python scripts/generate_quantum_search_data.py
```
This will produce `showcase/data/quantum_search_data.json`.

### Using the Agent
The agent can be invoked via the standard Agent Orchestrator or directly in Python:

```python
from core.agents.specialized.quantum_search_agent import QuantumSearchAgent

agent = QuantumSearchAgent(config={"agent_id": "avg_search_01"})
result = await agent.execute(task="hybrid_search", target_n=1e15)
print(result["pipeline"]["verification_status"])
```

## Theoretical Background
For a detailed quantitative analysis, refer to the whitepaper:
`docs/whitepapers/probabilistic_determinism_unstructured_search.md`

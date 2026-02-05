# Adam v26.0 Tutorials

Learn how to leverage the "System 2" reasoning engine for financial analysis.

## Tutorial 1: Running Your First Deep Dive Analysis

In this tutorial, you will use the **Fundamental Analyst Agent** to generate an investment memo.

### 1. Launch the CLI
Open your terminal and ensure your environment is active:
```bash
source .venv/bin/activate
python scripts/run_adam.py
```

### 2. Submit a Request
Enter the following command:
```text
User> Conduct a deep dive analysis on Apple (AAPL). Focus on the impact of the latest iPhone release on margins.
```

### 3. Observe the "Thinking" Process
Adam will now trigger the **Neuro-Symbolic Planner**. You will see logs indicating:
*   **Planning**: Decomposing the query into sub-tasks (e.g., Fetch 10-K, Analyze Segment Revenue, Check Competitor News).
*   **Execution**: Sub-agents (Swarm) fetching data.
*   **Synthesis**: The main engine drafting the report.

### 4. Review the Output
The system will output a structured markdown report. You can save this to a file or view it in the dashboard.

---

## Tutorial 2: Using the Crisis Simulator

Simulate macro-economic shocks to test portfolio resilience.

### 1. Access the Simulator
Open `showcase/index.html` and navigate to the **Crisis Simulator** tab.

### 2. Select a Scenario
Choose "Liquidity Shock (Jan 30)" from the dropdown menu. This scenario simulates a sudden drying up of repo market liquidity.

### 3. Adjust Parameters
*   **VIX Spike**: Set to 45.
*   **Repo Haircut**: Increase to 15%.

### 4. Run Simulation
Click "Execute Simulation". Watch as the **Risk Gauge** updates in real-time, showing the impact on LCR (Liquidity Coverage Ratio) and CET1 capital.

---

## Tutorial 3: Building a Custom Agent

(Advanced) Learn how to add a new specialist to the swarm.

1.  **Create the Agent File**: Create `core/agents/specialized/my_agent.py`.
2.  **Define the Class**: Inherit from `BaseAgent`.
3.  **Register the Tool**: Add your agent to `core/engine/neuro_symbolic_planner.py`.

*See `core/README.md` for detailed API documentation.*

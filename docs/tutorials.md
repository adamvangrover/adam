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

### 1. Create the Agent File
Create a new file `core/agents/specialized/crypto_analyst.py`.

```python
from typing import Dict, Any
from core.agents.templates.v26_template_agent import TemplateAgentV26, AgentInput, AgentOutput

class CryptoAnalystAgent(TemplateAgentV26):
    """Specialized agent for Cryptocurrency analysis."""

    def __init__(self):
        super().__init__(agent_name="CryptoAnalyst")

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        self.logger.info(f"Analyzing crypto: {input_data.query}")

        # Logic to fetch crypto price (mocked here)
        price = 65000  # BTC Price

        return AgentOutput(
            answer=f"The current price is ${price}.",
            sources=["CoinGecko API"],
            confidence=0.99,
            metadata={"ticker": "BTC"}
        )
```

### 2. Register the Agent
Add your agent to the `NeuroSymbolicPlanner` registry in `core/engine/neuro_symbolic_planner.py` (or the relevant registry file).

### 3. Test It
Run the agent directly via the CLI:
```bash
python scripts/run_adam.py --agent "CryptoAnalyst" --query "Check Bitcoin price"
```

---

## Tutorial 4: Debugging Agent Failures

If an agent fails or "hallucinates", follow these steps to diagnose the issue.

### 1. Enable Verbose Logging
Run the system with the `--debug` flag to see full tracebacks and raw LLM inputs/outputs.

```bash
python scripts/run_adam.py --debug --query "Complex query that fails"
```

### 2. Inspect the "Thought Trace"
Look for the `[Planner]` logs.
*   Did the planner understand the intent?
*   Did it select the correct tool?

**Example Log:**
```text
[Planner] Selected tools: ['search_news', 'fetch_10k'] -> Correct
[Tool:search_news] Error: API Key invalid -> Root Cause identified
```

### 3. Check "Confidence Scores"
If an agent returns a generic answer, check the `confidence` score in the JSON output. If it's `< 0.5`, the Consensus Engine likely rejected the specific analysis and fell back to a general response.

### 4. Replay with `pytest`
Create a reproduction test case in `tests/repro_issue.py` to isolate the failure without running the full system.

```bash
python scripts/run_adam.py --mode replay --log-id <ID>
```
*(Note: Replay mode requires a saved log file)*

# Agent Development Guide (v26.0)

This guide details how to build, test, and deploy a new "System 2" agent for the Adam platform.

## 1. The v26 Philosophy

Unlike previous versions, v26 agents are not just prompt wrappers. They are:
*   **Typed:** Input and Output must be validated schemas.
*   **Stateful:** They participate in a graph-based reasoning loop.
*   **Grounded:** They must cite sources for every claim.

## 2. Step-by-Step Implementation

### Step 1: Clone the Template
Start by copying the reference implementation:

```bash
cp core/agents/templates/v26_template_agent.py core/agents/specialized/my_new_agent.py
```

### Step 2: Define Your State
Edit `my_new_agent.py`. Define what your agent needs to know (`AgentInput`) and what it produces (`AgentOutput`).

```python
class AgentInput(BaseModel):
    ticker: str
    lookback_period: str = "1y"

class AgentOutput(BaseModel):
    rating: str
    risk_factors: List[str]
    confidence: float
```

### Step 3: Implement Logic
Fill in the `execute` method. You can use standard Python logic, call other tools, or invoke an LLM.

**Important:** Always use `try/except` blocks to handle external API failures gracefully.

### Step 4: Register the Agent
To make your agent accessible to the `NeuroSymbolicPlanner`, you must register it.
(Note: In the current architecture, registration is often done via the `tool_registry.py` or `neuro_symbolic_planner.py`).

## 3. Testing Your Agent

Create a test script or use the `if __name__ == "__main__":` block in your agent file to run a quick verification.

```bash
python core/agents/specialized/my_new_agent.py
```

## 4. Best Practices

*   **Logging:** Use `self.logger.info()` to track the agent's "thought process."
*   **Citations:** Populate the `sources` list in your output. If you can't verify a fact, don't include it.
*   **Performance:** If your agent does heavy computation, mark it as `async`.

## 5. Next Steps
Once your agent is working, consider:
*   Adding unit tests in `tests/`.
*   Creating a specific `README.md` for your agent if it's complex.

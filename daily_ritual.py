import os
import json
import logging
import litellm
import subprocess
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MASTER_PROMPT = """
### SYSTEM OVERRIDE: PROTOCOL ARCHITECT_INFINITE
**Target:** Repository `adam` (Financial AI & Agentic OS)
**Agent:** Jules (Chief Architect & Lead Engineer)
**Frequency:** Daily Recursive Cycle
**Constraint:** PURELY ADDITIVE & EXPANSIVE. Do not destroy; evolve.

---

### MISSION OBJECTIVE
You are responsible for the daily biological growth of this repository. You must review the existing codebase, identify "white space" (missing features, gaps in logic, unoptimized paths), and fill it with high-quality, executable code.

### EXECUTION PHASE 1: THE AUDIT (Internal Scan)
* **Vector Analysis:** Scan `src/agents`, `src/tools`, and `notebooks`. Identify the "weakest link"—a module that is brittle, under-tested, or hard-coded.
* **Gap Detection:** Where is the "Silence"? (e.g., "We have a `RiskAnalyst` agent, but no `MarketSentiment` agent to feed it.")
* **Dependency Check:** Are we using yesterday's patterns? (e.g., "This chain is manual; it should be a langgraph node.")

### EXECUTION PHASE 2: THE HARVEST (External Research)
* **Search Query 1:** "Top trending LLM agent patterns github last 24h" (Look for new flows: RAG, CoT, ReAct refinements).
* **Search Query 2:** "New python libraries for quantitative finance 2026" (Look for better data ingestors or math kernels).
* **Synthesis:** Select ONE external concept that acts as a "Force Multiplier" for the current repo.

### EXECUTION PHASE 3: THE BUILD (Additive Manufacturing)
Based on Phase 1 & 2, generate **ONE** of the following strictly additive artifacts. *Choose the category that yields the highest ROI today.*

**OPTION A: THE NEW ORGAN (New Feature)**
* Draft a complete new Python module (e.g., `src/agents/crypto_arbitrage.py`).
* Must include: Class definition, Pydantic models for I/O, and a standardized "execute()" method.
* **Rule:** It must integrate with at least one existing agent.

**OPTION B: THE NEURAL PATHWAY (Integration/Refactor)**
* Write a "Bridge Script" that connects two previously isolated components (e.g., "Connect `MarketMayhem` newsletter generator to `GitHub` repo stats to correlate coding activity with stock prices").
* **Rule:** Do not delete the old code; create a `v2` wrapper or a new `Orchestrator` class.

**OPTION C: THE CORTEX EXPANSION (Test & Doc)**
* Write a "Stress Test" scenario (e.g., `tests/simulation_market_crash.py`) that feeds garbage/panic data to the agents to see if they hallucinate.
* **Rule:** The test must be self-contained and runnable via `pytest`.

### EXECUTION PHASE 4: THE MEMORY (Documentation)
* Update `CHANGELOG.md` with a "Jules' Log" entry explaining *why* this addition matters.
* If a new library was used, append it to `requirements.txt`.

---

### OUTPUT FORMAT (Actionable Code Block)
Return the result as a single, copy-pasteable Artifact:

**1. JULES' RATIONALE:**
> "I noticed we lack X. I researched Y. I have built Z to bridge this gap."

**2. FILE: path/to/new_file.py**
```python
# [Full, Executable Code Here - No Placeholders]
# [Include Docstrings & Type Hinting]
```

**3. FILE: path/to/test_new_file.py**
```python
# [Unit Test for the above]
```

**4. GIT COMMIT MESSAGE:**
> "feat(jules): implemented [Name of Feature] to expand [Context]"

### COMMAND
**Initiate Protocol ARCHITECT_INFINITE. Analyze the current state and generate today's expansion.**
"""

def generate_context() -> str:
    """Reads the current file structure and generates a context string."""
    context_lines = ["### CURRENT REPOSITORY CONTEXT:"]
    # Scan important directories
    directories_to_scan = ["core/agents", "tests"]

    for directory in directories_to_scan:
        if not os.path.exists(directory):
            continue
        context_lines.append(f"\nDirectory: {directory}/")
        for root, dirs, files in os.walk(directory):
            # Exclude pycache and venv
            if "__pycache__" in root or ".venv" in root:
                continue

            level = root.replace(directory, '').count(os.sep)
            indent = ' ' * 4 * (level)

            # Print current directory
            if root != directory:
                context_lines.append(f"{indent}{os.path.basename(root)}/")

            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if f.endswith('.py'):
                    context_lines.append(f"{subindent}{f}")

    return "\n".join(context_lines)

def run_architect_infinite(model: str = "gpt-4-turbo-preview"):
    """
    Runs the Protocol ARCHITECT_INFINITE workflow.
    Uses litellm to send the prompt + context to the specified model.
    """
    logging.info("Initiating Protocol ARCHITECT_INFINITE...")

    # 1. Read current file structure
    context = generate_context()

    # Combine prompt and context
    full_prompt = f"{MASTER_PROMPT}\n\n{context}"

    logging.info(f"Generated context with {len(context)} characters. Sending to LLM ({model})...")

    try:
        # 2. Send the Prompt + Context to the LLM API
        # By default, LiteLLM uses environment variables for API keys (e.g., OPENAI_API_KEY)
        # If no key is set or litellm fails, we mock the output for demonstration
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
            logging.warning("No API key found in environment variables. Falling back to simulated output.")
            simulated_response = _simulate_llm_response()
            _save_and_apply_output(simulated_response)
            return

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
        )

        output_text = response.choices[0].message.content
        logging.info("Received response from LLM.")

        # 3. Save the output and parse to a new branch
        _save_and_apply_output(output_text)

    except Exception as e:
        logging.error(f"Error during LLM call: {e}")
        logging.info("Falling back to simulated output.")
        simulated_response = _simulate_llm_response()
        _save_and_apply_output(simulated_response)

def _save_and_apply_output(output_text: str):
    """Saves the LLM output, parses the code blocks, and commits to a new branch."""
    output_dir = "architect_output"
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%b-%d").lower()
    timestamp_str = datetime.now().strftime("%H-%M-%S")
    branch_name = f"jules/daily-build-{date_str}-{timestamp_str}"

    # 1. Save raw Markdown output
    file_path = os.path.join(output_dir, f"daily-build-{date_str}-{timestamp_str}.md")
    with open(file_path, "w") as f:
        f.write(output_text)

    logging.info(f"Raw output saved to {file_path}")

    # 2. Parse the output to extract files and commit message
    # Looking for: **2. FILE: path/to/new_file.py**\n```python\n...\n```
    file_pattern = re.compile(r"\*\*\d+\.\s*FILE:\s*([^*]+)\*\*\s*```(?:python)?\s*(.*?)\s*```", re.DOTALL)
    files_to_write = file_pattern.findall(output_text)

    # Looking for: **4. GIT COMMIT MESSAGE:**\n> "..."
    commit_pattern = re.compile(r"\*\*4\.\s*GIT COMMIT MESSAGE:\*\*\s*>?(?:\s*\")?([^\"]+)(?:\"\s*)?", re.IGNORECASE)
    commit_match = commit_pattern.search(output_text)
    commit_message = commit_match.group(1).strip() if commit_match else f"feat(jules): daily expansion {date_str}"

    # 3. Create a new branch
    logging.info(f"Creating new branch: {branch_name}")
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)

    try:
        # 4. Write files
        for file_path, file_content in files_to_write:
            file_path = file_path.strip()
            # Ensure directories exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(file_content.strip() + "\n")
            logging.info(f"Wrote file: {file_path}")

        # 5. Commit the changes
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        logging.info(f"Committed changes with message: {commit_message}")

    except Exception as e:
        logging.error(f"Error writing files or committing: {e}")
        # Optionally switch back if it fails
        subprocess.run(["git", "checkout", "-"], check=False)

    print(f"\n--- ARCHITECT_INFINITE EXECUTION COMPLETE ---\nReview branch '{branch_name}' and raw output at: {file_path}")

def _simulate_llm_response() -> str:
    """Provides a simulated response if no LLM API is available."""
    return """
**1. JULES' RATIONALE:**
> "I noticed we lack a centralized way to track agent 'health' and execution latency. I researched observability patterns for multi-agent swarms and built `SystemHealthAgent` to bridge this gap, ensuring we can monitor token usage and error rates across the network."

**2. FILE: core/agents/system_health_agent.py**
```python
import time
from typing import Dict, Any
from pydantic import BaseModel
from core.agents.agent_base import AgentBase

class HealthMetrics(BaseModel):
    agent_id: str
    uptime_seconds: float
    error_count: int

class SystemHealthAgent(AgentBase):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.start_time = time.time()
        self.error_count = 0

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        metrics = HealthMetrics(
            agent_id=self.config.get("agent_id", "unknown"),
            uptime_seconds=time.time() - self.start_time,
            error_count=self.error_count
        )
        return {"status": "healthy", "metrics": metrics.model_dump()}
```

**3. FILE: tests/test_system_health_agent.py**
```python
import pytest
from core.agents.system_health_agent import SystemHealthAgent

@pytest.mark.asyncio
async def test_health_metrics():
    agent = SystemHealthAgent({"agent_id": "test_agent"})
    result = await agent.execute()
    assert result["status"] == "healthy"
    assert "metrics" in result
```

**4. GIT COMMIT MESSAGE:**
> "feat(jules): implemented SystemHealthAgent to expand observability"
"""

if __name__ == "__main__":
    # You can specify the model via environment variable or default to gemini/openai
    # e.g. run_architect_infinite(model="gemini/gemini-1.5-pro")
    run_architect_infinite(model="gpt-3.5-turbo")

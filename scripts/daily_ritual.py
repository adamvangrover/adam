#!/usr/bin/env python3
"""
Protocol: ARCHITECT_INFINITE Wrapper Script
This script implements the automated daily cron job wrapper as requested.
"""
import os
import sys
import subprocess
import datetime
from litellm import completion

# The Master Prompt provided by the user
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

def generate_context_string():
    """Generates a context string by walking relevant directories."""
    # Based on the prompt, it targets src/agents, src/tools, notebooks.
    # In the adam repo, agents are mainly in core/agents and core/v30_architecture/python_intelligence/agents
    directories_to_scan = [
        "core/agents",
        "core/v30_architecture/python_intelligence/agents",
        "core/tools"
    ]

    context_str = "### CURRENT REPOSITORY STATE\n"

    for base_dir in directories_to_scan:
        if not os.path.exists(base_dir):
            continue

        context_str += f"\n--- Files in {base_dir} ---\n"
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith('.py') or f.endswith('.md'):
                    context_str += f"- {os.path.join(root, f)}\n"

    return context_str

def create_new_branch():
    """Creates a new branch for the daily build."""
    date_str = datetime.datetime.now().strftime("%b-%d").lower()
    branch_name = f"jules/daily-build-{date_str}"
    print(f"Creating new branch: {branch_name}")
    try:
        # Create and checkout new branch
        subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
        return branch_name
    except subprocess.CalledProcessError as e:
        print(f"Error creating branch: {e.stderr.decode()}")
        # Fallback to current branch
        return None

def main():
    print("Initiating Protocol ARCHITECT_INFINITE wrapper...")

    # 1. Generate Context
    print("Generating context from file structure...")
    context = generate_context_string()

    # 2. Combine Prompt + Context
    full_prompt = f"{MASTER_PROMPT}\n\n{context}"

    # Create branch
    branch = create_new_branch()

    # 3. Call LLM
    print("Sending prompt to LLM (using LiteLLM)...")
    # For a real run, you need OPENAI_API_KEY or similar environment variable
    try:
        response = completion(
            model="gpt-4", # Defaulting to a strong model
            messages=[{"role": "user", "content": full_prompt}]
        )
        output_content = response.choices[0].message.content

        print("\n--- LLM RESPONSE RECEIVED ---")

        # 4. Save Output (In a real scenario, this script would parse the markdown
        # and create files automatically. For now, we save it as a pending commit artifact)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        output_file = f"daily_architect_output_{date_str}.md"
        with open(output_file, 'w') as f:
            f.write(output_content)
        print(f"Output saved to {output_file}. Manual review and execution recommended.")

    except Exception as e:
        print(f"Error calling LLM: {e}")
        print("\nNote: You need to set your LLM provider API key (e.g., OPENAI_API_KEY) in your environment to use this script.")

if __name__ == "__main__":
    main()

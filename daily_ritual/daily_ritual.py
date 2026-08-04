import os
import re
import datetime
from datetime import timezone
import subprocess
from openai import OpenAI

def get_master_prompt() -> str:
    current_day = datetime.datetime.now(timezone.utc).strftime("%A")

    daily_tasks = {
        "Monday": "The Pruning (Deterministic Integrity)\n*Goal: Remove bloat and non-deterministic \"ghost\" logic to simplify the context window.*\n\n> \"Analyze this code. Remove all dead code, unused imports, and legacy logic that obfuscates the data flow. Specifically, identify any 'black box' logic that lacks clear input/output types and simplify it. Return the cleaned code and a summary of what was removed to reduce cognitive load on the system.\"",
        "Tuesday": "The Refactor (Architecture & Modularity)\n*Goal: Decouple data ingestion from logic to facilitate horizontal/vertical scalability.*\n\n> \"Refactor this module into a clean, reusable architecture. Abstract logic into modular components, prioritizing the separation of raw data ingestion and model-based processing. Apply structural patterns that ensure this module can be independently tested and easily integrated into the broader Adam framework.\"",
        "Wednesday": "The PDIL Optimizer (Performance & Safety)\n*Goal: Audit the probabilistic-to-deterministic transition.*\n\n> \"Audit this code for bottlenecks and potential 'stochastic drift'—where model output might violate system invariants. Implement strict schema validation (e.g., Pydantic) to secure the PDIL bridge. Optimize performance for high-throughput and ensure robust error handling that maintains state consistency.\"",
        "Thursday": "The Modernizer (Syntactic & Type Safety)\n*Goal: Upgrade to the latest idioms and ensure rigid type checking.*\n\n> \"Modernize this codebase to align with the latest language standards (e.g., Python 3.12+ features, strict typing, async patterns). Replace any anti-patterns that hinder static analysis or AI-readability. Ensure that all type hints are explicit, supporting high-assurance engineering.\"",
        "Friday": "The Innovator (Agentic Enhancement)\n*Goal: Inject autonomous logic that adheres to the established framework.*\n\n> \"Analyze this code's core utility. Propose 3 high-value enhancements (e.g., autonomous self-healing, recursive RAG pipelines, or smarter parsing). Implement the most high-value feature, ensuring it includes a 'ProvenanceHeader' to maintain W3C PROV-O compliance and trace decision-making.\"",
        "Saturday": "The Documenter (Context-First Documentation)\n*Goal: Make the code 'AI-native' for future context windows.*\n\n> \"Read through this updated code. Generate high-density docstrings that prioritize **Provenance Metadata** (Input/Output schemas, intended state, and derivation paths). Create an 'Architecture & Usage' snippet for the README that allows another AI to fully reconstruct the module's behavior from its documentation.\"",
        "Sunday": "The Validator (Deterministic Test Suite)\n*Goal: Secure the PDIL bridge with property-based testing.*\n\n> \"Generate a comprehensive test suite using a modern framework (e.g., `pytest` with `hypothesis`). Focus on property-based testing to stress-test the probabilistic inputs against deterministic boundaries. Ensure all external calls are mocked and that failure states explicitly log the provenance of the erroneous input.\""
    }

    task_description = daily_tasks.get(current_day, daily_tasks["Monday"])

    return f"""### SYSTEM OVERRIDE: PROTOCOL ARCHITECT_INFINITE
**Target:** Repository `adam` (Financial AI & Agentic OS)
**Agent:** Jules (Chief Architect & Lead Engineer)
**Frequency:** Daily Recursive Cycle
**Constraint:** PURELY ADDITIVE & EXPANSIVE. Do not destroy; evolve.

---

### MISSION OBJECTIVE
You are responsible for the daily biological growth of this repository. You must review the existing codebase, identify "white space" (missing features, gaps in logic, unoptimized paths), and fill it with high-quality, executable code.

### PDIL AND W3C PROV-O INTEGRATION
Every upgrade must be aligned with the Adam architecture.
- Ensure that the Probabilistic-to-Deterministic Integration Layer (PDIL) is secure and well-defined.
- All additions must maintain W3C PROV-O compliance, tracing decision-making via `ProvenanceHeader` and structured logging.

### TODAY'S TASK: {current_day} - {task_description}

### EXECUTION PHASE 1: THE AUDIT (Internal Scan)
* **Vector Analysis:** Scan `src/agents`, `src/tools`, and `notebooks`. Identify the "weakest link"—a module that is brittle, under-tested, or hard-coded.
* **Gap Detection:** Where is the "Silence"? (e.g., "We have a `RiskAnalyst` agent, but no `MarketSentiment` agent to feed it.")
* **Dependency Check:** Are we using yesterday's patterns? (e.g., "This chain is manual; it should be a langgraph node.")

### EXECUTION PHASE 2: THE HARVEST (External Research)
* **Search Query 1:** "Top trending LLM agent patterns github last 24h" (Look for new flows: RAG, CoT, ReAct refinements).
* **Search Query 2:** "New python libraries for quantitative finance 2026" (Look for better data ingestors or math kernels).
* **Synthesis:** Select ONE external concept that acts as a "Force Multiplier" for the current repo.

### EXECUTION PHASE 3: THE BUILD (Additive Manufacturing)
Based on Phase 1 & 2 and Today's Task, generate **ONE** strictly additive artifact.
* Focus on reusability. If you find yourself writing a new function, check if it can be elevated to the `core_types.py` level for horizontal use.

### EXECUTION PHASE 4: THE MEMORY (Documentation)
* Update `CHANGELOG.md` with a "Jules' Log" entry explaining *why* this addition matters.
* If a new library was used, append it to `requirements/base.txt`.

---

### OUTPUT FORMAT (Actionable Code Block)
Return the result as a single, copy-pasteable Artifact:

**1. JULES' RATIONALE:**
> "I noticed we lack X. I researched Y. I have built Z to bridge this gap."

**1.5. OBSERVED DRIFT:**
> "[Explicitly state if the refactor changes the module's behavior significantly, flagging it for manual review to ensure collective determinism is not compromised.]"

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
    """Generates a list of files in the project via os.walk"""
    output = []
    # Only walk important source directories to avoid exploding the context window
    target_dirs = ["core", "src", "tests", "notebooks"]
    for d in target_dirs:
        if os.path.exists(d):
            for root, dirs, files in os.walk(d):
                # Skip caches
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for file in files:
                    if file.endswith(".py") or file.endswith(".md"):
                        path = os.path.join(root, file)
                        output.append(path)

    return "Repository File Structure (Context):\n" + "\n".join(output)

def call_llm(context_str: str) -> str:
    """Calls the Gemini API (via litellm) with the prompt and context."""
    import litellm
    import os

    master_prompt = get_master_prompt()

    messages = [
        {"role": "system", "content": master_prompt},
        {"role": "user", "content": context_str}
    ]

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            response = litellm.completion(
                model="gemini/gemini-pro",
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Gemini LLM: {e}")
    else:
        # Fallback to OpenAI API logic if OpenAI is configured instead
        openai_key = os.environ.get("OPENAI_API_KEY", "dummy")
        client = OpenAI(api_key=openai_key)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI LLM: {e}")

    # Return fallback mock if all API calls fail or no keys are set
    print("Using fallback mock response.")
    date_str = datetime.datetime.now().strftime("%Y-%b-%d").lower()
    return f"""**1. JULES' RATIONALE:**
> "I noticed we lack a structured way to monitor yield farming opportunities across decentralized finance protocols. I researched autonomous yield optimization patterns. I have built YieldFarmingAgent to bridge this gap, integrating with the existing RiskAssessmentAgent to ensure safe capital allocation."

**1.5. OBSERVED DRIFT:**
> "No drift observed. The changes are purely additive and maintain the deterministic integrity of the system."

**2. FILE: core/agents/yield_farming_agent.py**
```python
import logging
from typing import Any, Dict, Optional

from core.schemas.agent_schema import AgentInput, AgentOutput
from core.agents.agent_base import AgentBase

logger = logging.getLogger(__name__)

class YieldFarmingAgent(AgentBase):
    \"\"\"
    The Yield Farming Agent scans DeFi protocols for high-yield opportunities,
    assesses the associated risks (e.g., impermanent loss, smart contract risk),
    and recommends capital allocation strategies.
    \"\"\"

    def __init__(self, config: Optional[Dict[str, Any]] = None, kernel: Optional[Any] = None):
        if config is None:
            config = {{"name": "YieldFarmingAgent"}}
        super().__init__(config, kernel=kernel)
        self.supported_protocols = ["Aave", "Compound", "Curve", "Uniswap"]

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        query = input_data.query.lower()
        logger.info(f"YieldFarmingAgent analyzing query: {{query}}")

        if not query:
            return AgentOutput(
                answer="No query provided for yield farming analysis.",
                confidence=0.0,
                metadata={{"status": "error"}}
            )

        # Simulated logic for finding yields
        recommended_pool = "Curve 3pool"
        estimated_apy = 4.5
        risk_level = "Low"

        if "high risk" in query:
            recommended_pool = "Uniswap V3 ETH/USDC (Narrow Range)"
            estimated_apy = 25.0
            risk_level = "High"

        answer = f"Recommended allocation: {{recommended_pool}} with an estimated APY of {{estimated_apy}}% (Risk: {{risk_level}})."

        return AgentOutput(
            answer=answer,
            confidence=0.85,
            metadata={{
                "recommended_pool": recommended_pool,
                "estimated_apy": estimated_apy,
                "risk_level": risk_level,
                "protocols_scanned": self.supported_protocols
            }}
        )
```

**3. FILE: tests/test_yield_farming_agent.py**

```python
import pytest
from core.schemas.agent_schema import AgentInput
from core.agents.yield_farming_agent import YieldFarmingAgent

@pytest.mark.asyncio
async def test_yield_farming_agent_low_risk():
    agent = YieldFarmingAgent(config={{'llm': {{'provider': 'mock', 'model': 'mock'}}}})
    input_data = AgentInput(query="Find me a stable yield", context={{}})
    result = await agent.execute(input_data)

    assert result.metadata["recommended_pool"] == "Curve 3pool"
    assert result.metadata["risk_level"] == "Low"

@pytest.mark.asyncio
async def test_yield_farming_agent_high_risk():
    agent = YieldFarmingAgent(config={{'llm': {{'provider': 'mock', 'model': 'mock'}}}})
    input_data = AgentInput(query="Find me a high risk yield", context={{}})
    result = await agent.execute(input_data)

    assert result.metadata["recommended_pool"] == "Uniswap V3 ETH/USDC (Narrow Range)"
    assert result.metadata["risk_level"] == "High"
```

**4. GIT COMMIT MESSAGE:**

> "feat(jules): implemented Yield Farming Agent to expand DeFi capabilities"
"""

def parse_and_apply_expansion(response_text: str):
    print("Parsing LLM response...")

    # 1. Extract Rationale
    rationale_match = re.search(r"\*\*1\. JULES' RATIONALE:\*\*\s*>(.*?)(?=\*\*1\.5\. OBSERVED DRIFT:|\*\*2\. FILE:|\Z)", response_text, re.DOTALL | re.IGNORECASE)
    rationale = rationale_match.group(1).strip() if rationale_match else "No rationale provided."

    # 1.5. Extract Observed Drift
    drift_match = re.search(r"\*\*1\.5\. OBSERVED DRIFT:\*\*\s*>(.*?)(?=\*\*2\. FILE:|\Z)", response_text, re.DOTALL | re.IGNORECASE)
    drift = drift_match.group(1).strip() if drift_match else "No drift observed."

    # 2. Extract Files
    file_matches = re.finditer(r"\*\*(?:\d\.)?\s*FILE:\s*([^*]+)\*\*\s*```(?:python)?\n(.*?)\n```", response_text, re.DOTALL | re.IGNORECASE)

    added_files = []
    # Artifacts will generate within the daily_ritual directory.
    # To avoid `daily_ritual/daily_ritual` when running from the directory itself,
    # we determine if we are already inside the daily_ritual directory.
    base_dir = "." if os.path.basename(os.getcwd()) == "daily_ritual" else "daily_ritual"
    os.makedirs(base_dir, exist_ok=True)

    for match in file_matches:
        filepath = match.group(1).strip()
        content = match.group(2)

        target_path = os.path.join(base_dir, filepath)

        # Ensure directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        with open(target_path, 'w') as f:
            f.write(content)
        added_files.append(target_path)
        print(f"Created file: {target_path}")

    # 3. Extract Commit Message
    commit_match = re.search(r"\*\*(?:\d\.)?\s*GIT COMMIT MESSAGE:\*\*\s*>(.*?)(?=\Z)", response_text, re.DOTALL | re.IGNORECASE)
    commit_msg = commit_match.group(1).strip().strip('"') if commit_match else "feat(jules): implemented new architect expansion"

    return rationale, drift, added_files, commit_msg

def update_changelog(rationale: str, drift: str, added_files: list):
    date_str = datetime.datetime.now().strftime("%Y-%b-%d").lower()

    new_entry = f"## [{date_str}] - Protocol ARCHITECT_INFINITE Expansion\n\n"
    new_entry += "### Jules' Log\n"
    new_entry += f"> {rationale}\n\n"
    new_entry += "### Observed Drift\n"
    new_entry += f"> {drift}\n\n"
    new_entry += "### Added\n"
    for file in added_files:
        new_entry += f"- Created/Updated `{file}`\n"
    new_entry += "\n\n"

    base_dir = "." if os.path.basename(os.getcwd()) == "daily_ritual" else "daily_ritual"
    os.makedirs(base_dir, exist_ok=True)
    changelog_path = os.path.join(base_dir, "CHANGELOG.md")

    # Prepend to CHANGELOG.md right after the main header
    if os.path.exists(changelog_path):
        with open(changelog_path, 'r') as f:
            content = f.read()

        header_match = re.search(r"^# Changelog\s*\n", content, re.MULTILINE)
        if header_match:
            insert_pos = header_match.end()
            new_content = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]
        else:
            new_content = "# Changelog\n\n" + new_entry + content

        with open(changelog_path, 'w') as f:
            f.write(new_content)
        print("Updated CHANGELOG.md")
    else:
        with open(changelog_path, 'w') as f:
            f.write("# Changelog\n\n" + new_entry)
        print("Created CHANGELOG.md")

def create_branch_and_commit(commit_msg: str, added_files: list):
    date_str = datetime.datetime.now().strftime("%b-%d").lower()
    branch_name = f"jules/daily-build-{date_str}"

    try:
        # Avoid checking out again if we are already on the branch
        current_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
        if current_branch != branch_name:
            # Checkout new branch
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            print(f"Checked out new branch: {branch_name}")

        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        print(f"Committed changes with message: {commit_msg}")

    except subprocess.CalledProcessError as e:
        print(f"Git operations failed: {e}")

def main():
    print("Initiating Protocol ARCHITECT_INFINITE...")

    # 1. Read context
    print("Scanning repository structure...")
    context_str = generate_context_string()

    # 2. Call LLM
    print("Calling LLM API...")
    response_text = call_llm(context_str)

    # 3. Apply changes
    rationale, drift, added_files, commit_msg = parse_and_apply_expansion(response_text)

    # 4. Update memory/docs
    update_changelog(rationale, drift, added_files)

    # 5. Git operations
    create_branch_and_commit(commit_msg, added_files)

    print("ARCHITECT_INFINITE cycle complete.")

if __name__ == "__main__":
    main()

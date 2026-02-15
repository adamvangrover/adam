
import os
import re

def audit_repo():
    print("### SYSTEM OVERRIDE: PROTOCOL ARCHITECT_INFINITE")
    print("**Target:** Repository `adam` (Financial AI & Agentic OS)")
    print("**Agent:** Jules (Chief Architect & Lead Engineer)")
    print("**Frequency:** Daily Recursive Cycle")
    print("**Constraint:** PURELY ADDITIVE & EXPANSIVE. Do not destroy; evolve.\n")
    print("---")
    print("### EXECUTION PHASE 1: THE AUDIT (Internal Scan)")

    agents_dir = "core/agents"
    tests_dir = "tests"

    print(f"* **Scanning** `{agents_dir}`...")

    agent_files = [f for f in os.listdir(agents_dir) if f.endswith(".py") and f != "__init__.py"]

    weak_links = []

    for agent_file in agent_files:
        agent_path = os.path.join(agents_dir, agent_file)

        # Check size
        with open(agent_path, "r") as f:
            content = f.read()
            lines = content.splitlines()
            loc = len(lines)
            todos = len(re.findall(r"#\s*TODO", content, re.IGNORECASE))

        # Check test
        test_file = "test_" + agent_file
        test_path = os.path.join(tests_dir, test_file)
        has_test = os.path.exists(test_path)

        if not has_test or loc < 50 or todos > 0:
            weak_links.append({
                "file": agent_file,
                "loc": loc,
                "has_test": has_test,
                "todos": todos
            })

    print(f"* **Found {len(weak_links)} weak links.**")
    for link in weak_links[:5]: # Show top 5
        print(f"  - `{link['file']}`: LOC={link['loc']}, Test={link['has_test']}, TODOs={link['todos']}")

    print("\n### EXECUTION PHASE 2: THE HARVEST (External Research)")
    print("* **Search Query 1:** \"Top trending LLM agent patterns github last 24h\"")
    print("* **Search Query 2:** \"New python libraries for quantitative finance 2026\"")

    print("\n### EXECUTION PHASE 3: THE BUILD (Additive Manufacturing)")
    print("Based on Phase 1 & 2, generate **ONE** of the following strictly additive artifacts.")
    print("**OPTION A:** New Feature (e.g. `src/agents/crypto_arbitrage.py`)")
    print("**OPTION B:** Integration/Refactor (e.g. Bridge Script)")
    print("**OPTION C:** Cortex Expansion (Test & Doc)")

    print("\n### EXECUTION PHASE 4: THE MEMORY (Documentation)")
    print("* Update `CHANGELOG.md`")
    print("* Update `requirements.txt` if needed")

if __name__ == "__main__":
    audit_repo()

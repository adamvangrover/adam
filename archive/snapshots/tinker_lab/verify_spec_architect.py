# verify_spec_architect.py
import asyncio
import sys
from core.agents.developer_swarm.spec_architect_agent import SpecArchitectAgent

async def run_checks():
    print("Initializing SpecArchitectAgent...")
    try:
        config = {"model": "gpt-4", "temperature": 0.0}
        agent = SpecArchitectAgent(config)
    except Exception as e:
        print(f"FAILED to initialize agent: {e}")
        return

    print("Agent initialized.")

    goal = "Add Two-Factor Authentication"
    print(f"Executing goal: {goal}")
    try:
        spec_content = await agent.execute(goal)
        print("Spec generated successfully.")
    except Exception as e:
        print(f"FAILED to generate spec: {e}")
        return

    # Verify content
    checks = [
        "# Spec: Add Two-Factor Authentication",
        "## 1. Overview & Objectives",
        "## 2. Technical Context",
        "## 3. Implementation Plan",
        "## 4. Commands & Development",
        "## 5. Verification & Testing Strategy",
        "## 6. Constraints & Boundaries",
        "✅ **Always:**",
        "🚫 **Never:**"
    ]

    failed_checks = []
    for check in checks:
        if check not in spec_content:
            failed_checks.append(check)

    if failed_checks:
        print("FAILED checks:")
        for fc in failed_checks:
            print(f"  Missing: {fc}")
    else:
        print("PASSED: All spec sections present.")

    # Check context handling
    print("\nTesting context handling...")
    context = ["src/db/models.py"]
    spec_content = await agent.execute(goal, context_files=context)
    if "Analyzed context from: src/db/models.py" in spec_content:
        print("PASSED: Context files acknowledged.")
    else:
        print("FAILED: Context files not acknowledged.")

if __name__ == "__main__":
    asyncio.run(run_checks())

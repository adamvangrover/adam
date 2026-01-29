import json
import asyncio
import os
import sys
# Ensure core is in path if run from subdirectory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from core.engine.swarm.pheromone_board import PheromoneBoard
from core.engine.swarm.neuro_worker import NeuroQuantumWorker

async def run_benchmark():
    print("--- Starting Adam Neuro-Quantum Assessment Benchmark ---")

    # Load questions
    with open(os.path.join(os.path.dirname(__file__), "adam_assessment_set.json"), "r") as f:
        questions = json.load(f)

    board = PheromoneBoard()
    worker = NeuroQuantumWorker(board)

    results_summary = {"passed": 0, "failed": 0, "details": []}

    for q in questions:
        print(f"Running Question ID: {q['id']} ({q['category']})")

        task = {
            "id": q['id'],
            "type": "SIMULATION",
            "context": q['prompt'],
            "steps": 5
        }

        await worker.execute_task(task)

        # Get result
        res_list = await board.sniff("NEURO_RESULT")
        # Filter for this task id
        res = next((r for r in res_list if r.data["task_id"] == q['id']), None)

        if not res:
            print(f"  [FAIL] No result found for {q['id']}")
            results_summary["failed"] += 1
            continue

        holistic_state = res.data["holistic_state"]
        passed = True
        reasons = []

        # Verify FIBO
        if "expected_fibo" in q:
            actual_fibo = holistic_state.get("fibo_concepts", [])
            for expected in q["expected_fibo"]:
                if expected not in actual_fibo:
                    passed = False
                    reasons.append(f"Missing FIBO concept: {expected}")

        # Verify Regime
        if "expected_regime" in q:
            actual_regime = holistic_state.get("market_regime", "Unknown")
            if actual_regime != q["expected_regime"]:
                passed = False
                reasons.append(f"Regime mismatch. Expected {q['expected_regime']}, got {actual_regime}")

        # Verify Entropy Range
        if "expected_entropy_min" in q:
            entropy = holistic_state.get("quantum_entropy", 0.0)
            if not (q["expected_entropy_min"] <= entropy <= q["expected_entropy_max"]):
                passed = False # Not strictly failing since mocking is random, but noting it
                reasons.append(f"Entropy {entropy} out of range")

        if passed:
            print("  [PASS]")
            results_summary["passed"] += 1
        else:
            print(f"  [FAIL] Reasons: {', '.join(reasons)}")
            results_summary["failed"] += 1
            results_summary["details"].append({"id": q['id'], "reasons": reasons})

        # Cleanup board for next iteration? Not strictly necessary as sniff filters,
        # but prevents buildup. In this simple script we assume unique IDs.

    print("\n--- Assessment Complete ---")
    print(f"Total Passed: {results_summary['passed']}")
    print(f"Total Failed: {results_summary['failed']}")
    if results_summary["failed"] > 0:
        print("Failures:")
        for fail in results_summary["details"]:
            print(f" - {fail['id']}: {fail['reasons']}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())

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
    questions_path = os.path.join(os.path.dirname(__file__), "adam_assessment_set.json")
    try:
        with open(questions_path, "r") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Assessment set not found at {questions_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {questions_path}")
        sys.exit(1)

    board = PheromoneBoard()
    worker = NeuroQuantumWorker(board)

    results_summary = {"passed": 0, "failed": 0, "results": []}

    print(f"{'ID':<15} | {'Category':<25} | {'Status':<10}")
    print("-" * 60)

    for q in questions:
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

        passed = True
        reasons = []

        if not res:
            passed = False
            reasons.append("No result found (timeout/error)")
        else:
            holistic_state = res.data["holistic_state"]

            # Verify FIBO
            if "expected_fibo" in q:
                actual_fibo = holistic_state.get("fibo_concepts", [])
                for expected in q["expected_fibo"]:
                    if expected not in actual_fibo:
                        passed = False
                        reasons.append(f"Missing FIBO: {expected}")

            # Verify Regime
            if "expected_regime" in q:
                actual_regime = holistic_state.get("market_regime", "Unknown")
                if actual_regime != q["expected_regime"]:
                    passed = False
                    reasons.append(f"Regime mismatch: exp {q['expected_regime']}, got {actual_regime}")

            # Verify Entropy Range
            if "expected_entropy_min" in q:
                entropy = holistic_state.get("quantum_entropy", 0.0)
                if not (q["expected_entropy_min"] <= entropy <= q["expected_entropy_max"]):
                    passed = False
                    reasons.append(f"Entropy {entropy:.2f} out of range")

        status_str = "PASS" if passed else "FAIL"
        print(f"{q['id']:<15} | {q['category']:<25} | {status_str:<10}")

        if passed:
            results_summary["passed"] += 1
        else:
            results_summary["failed"] += 1

        results_summary["results"].append({
            "id": q['id'],
            "passed": passed,
            "reasons": reasons
        })

    print("-" * 60)
    print("\n--- Assessment Summary ---")
    print(f"Total Tests:  {len(questions)}")
    print(f"Passed:       {results_summary['passed']}")
    print(f"Failed:       {results_summary['failed']}")

    if results_summary["failed"] > 0:
        print("\nFailure Details:")
        print(f"{'ID':<15} | {'Reasons'}")
        print("-" * 80)
        for res in results_summary["results"]:
            if not res["passed"]:
                reason_str = "; ".join(res["reasons"])
                print(f"{res['id']:<15} | {reason_str}")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(run_benchmark())

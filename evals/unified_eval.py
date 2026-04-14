import json
import os
import subprocess
import time
from datetime import datetime

def run_command(cmd):
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout, "error": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"status": "failed", "output": e.stdout, "error": e.stderr}

def main():
    print("Starting Unified Evaluation Harness...")
    results = {
        "timestamp": datetime.now().isoformat(),
        "evaluations": {}
    }

    # 1. Crisis Simulation
    print("Running Crisis Simulation Evaluation...")
    start_time = time.time()
    crisis_res = run_command(["uv", "run", "pytest", "evals/eval_crisis_sim.py"])
    results["evaluations"]["crisis_sim"] = {
        "duration_seconds": round(time.time() - start_time, 2),
        "status": crisis_res["status"],
        "logs": crisis_res["output"] + "\n" + crisis_res["error"]
    }

    # 2. RAG Pipeline
    print("Running RAG Pipeline Evaluation...")
    start_time = time.time()
    rag_res = run_command(["uv", "run", "python", "evals/eval_rag_pipeline.py"])
    results["evaluations"]["rag_pipeline"] = {
        "duration_seconds": round(time.time() - start_time, 2),
        "status": rag_res["status"],
        "logs": rag_res["output"] + "\n" + rag_res["error"]
    }

    # 3. LLM Judge Eval
    print("Running LLM Judge Evaluation...")
    start_time = time.time()
    judge_res = run_command(["uv", "run", "python", "evals/run.py"])
    results["evaluations"]["llm_judge"] = {
        "duration_seconds": round(time.time() - start_time, 2),
        "status": judge_res["status"],
        "logs": judge_res["output"] + "\n" + judge_res["error"]
    }

    # Output JSON Report
    report_path = "evals/data/unified_eval_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Unified Evaluation Complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()

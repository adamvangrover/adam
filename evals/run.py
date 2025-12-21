import argparse
import json
import logging
from typing import Dict, List

try:
    from .graders.llm_judge import grade_answer
except ImportError:
    # Handle relative import when running as script
    from graders.llm_judge import grade_answer

logger = logging.getLogger(__name__)

def load_dataset(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return json.load(f)

def run_agent_mock(question: str, ticker: str) -> str:
    """
    Simulates running the agent on a specific question.
    In production, this would invoke 'app.invoke(...)'.
    """
    if "Net Debt" in question:
        return "Based on the 2023 10-K, 3M's Net Debt was approximately $10.5 Billion."
    elif "Net Leverage" in question:
        return "The Net Leverage Ratio covenant is set at 4.50x."
    return "I could not find that information."

def run_evals(dataset_path: str):
    print(f"--- Starting Evaluation on {dataset_path} ---")
    data = load_dataset(dataset_path)

    scores = []
    for item in data:
        q = item["question"]
        gold = item["answer"]
        ticker = item["ticker"]

        print(f"Evaluating: {q}")
        agent_ans = run_agent_mock(q, ticker)
        score = grade_answer(q, agent_ans, gold)

        scores.append(score)
        print(f"  Result: {score}/1.0")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"--- Evaluation Complete. Average Score: {avg_score:.2f} ---")

    if avg_score < 0.9:
        print("FAIL: Quality threshold not met.")
    else:
        print("PASS: Quality threshold met.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="evals/data/finance_bench.json")
    args = parser.parse_args()

    run_evals(args.data)

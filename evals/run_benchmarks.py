import json
import asyncio
import os
from typing import List, Dict
import re

# Mocking Agent and LLM-as-a-Judge for the benchmark script
# In a real scenario, import the actual agent and LLM client

async def run_agent(question: str) -> str:
    """
    Simulates the agent's response to a question.
    In reality, this would invoke the CyclicalReasoningGraph or similar.
    """
    print(f"Agent answering: {question}")
    # Mock responses based on the question
    if "Net Income" in question:
        return "Based on the 2023 10-K, AAPL's Net Income was $96.995 billion."
    elif "Debt/EBITDA" in question:
        return "The calculated Debt/EBITDA is 3.5x, which is below the threshold."
    elif "leverage ratio" in question:
        return "The Credit Agreement specifies a maximum Consolidated Leverage Ratio of 4.50 to 1.00."
    else:
        return "I cannot answer this question based on available data."

def llm_grader(question: str, prediction: str, ground_truth: str) -> float:
    """
    Simulates an LLM-as-a-Judge.
    Returns a score between 0.0 and 1.0.
    """
    # Simple heuristic for the mock
    # Check if numbers in ground_truth are present in prediction

    # Extract numbers from ground truth
    gt_nums = re.findall(r"[-+]?\d*\.\d+|\d+", ground_truth)

    matches = 0
    for num in gt_nums:
        if num in prediction:
            matches += 1

    if not gt_nums:
        # If no numbers, do basic keyword matching or length check (mock)
        return 1.0 if len(prediction) > 10 else 0.0

    score = matches / len(gt_nums)
    return float(score)

async def run_benchmarks():
    print("Starting Benchmark Run...")

    # Load Golden Set
    filepath = "data/finance_bench.json"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    with open(filepath, "r") as f:
        benchmarks = json.load(f)

    results = []

    for item in benchmarks:
        question = item["question"]
        ground_truth = item["answer"]

        # Run Agent
        prediction = await run_agent(question)

        # Grade
        score = llm_grader(question, prediction, ground_truth)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "score": score
        })

        print(f"Q: {question}\nScore: {score}\n")

    # Calculate Average
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"Benchmark Complete. Average Score: {avg_score:.2f}")

if __name__ == "__main__":
    asyncio.run(run_benchmarks())

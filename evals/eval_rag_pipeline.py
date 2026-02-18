import sys
import os
import asyncio
import logging
from typing import Dict, List, Any

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generate_mock_10ks import MOCK_DATA, main as generate_mock_data
from scripts.run_credit_memo_rag import CreditMemoRAGPipeline

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("RAGEval")

async def run_eval():
    print("--- Starting RAG Pipeline Evaluation ---")

    # 1. Ensure Data Exists
    print("Generating/Verifying Mock Data...")
    generate_mock_data()

    results = []

    # 2. Iterate and Evaluate
    for item in MOCK_DATA:
        ticker = item["ticker"]
        filename = f"data/10k_sample_{ticker.lower()}.txt"

        print(f"\nEvaluating {ticker}...")

        pipeline = CreditMemoRAGPipeline(ticker)
        spread, memo = await pipeline.run(filename)

        if not spread:
            print(f"FAILED: Could not extract data for {ticker}")
            continue

        extracted = spread["metrics"]

        # Compare
        # Revenue
        actual_rev = item["revenue"]
        extracted_rev = extracted.get("Revenue", 0)
        rev_err = abs(extracted_rev - actual_rev) / actual_rev if actual_rev else 0

        # EBITDA (Mock 10K doesn't explicitly state EBITDA, but the pipeline estimates it.
        # However, the mock data has 'op_income'. The pipeline estimates EBITDA as Op Income * 1.10)
        # So we should compare extracted EBITDA to (item['op_income'] * 1.10) approximately?
        # Or better, just compare what we have control over.
        # The pipeline looks for "total net sales", "net income", "total assets", "total liabilities", "total debt", "cash".

        # Let's check Net Income
        actual_ni = item["net_income"]
        extracted_ni = extracted.get("Net Income", 0)
        ni_err = abs(extracted_ni - actual_ni) / abs(actual_ni) if actual_ni else 0

        # Total Debt
        actual_debt = item["debt"]
        extracted_debt = extracted.get("Total Debt", 0)
        debt_err = abs(extracted_debt - actual_debt) / abs(actual_debt) if actual_debt else 0

        # Cash
        actual_cash = item["cash"]
        extracted_cash = extracted.get("Cash", 0)
        cash_err = abs(extracted_cash - actual_cash) / abs(actual_cash) if actual_cash else 0

        score = 0
        checks = 0

        # Revenue Check (allow 1% error)
        if rev_err < 0.01: score += 1
        checks += 1

        # Net Income Check (allow 1% error)
        if ni_err < 0.01: score += 1
        checks += 1

        # Debt Check (allow 1% error)
        if debt_err < 0.01: score += 1
        checks += 1

        # Cash Check (allow 1% error)
        if cash_err < 0.01: score += 1
        checks += 1

        accuracy = score / checks

        print(f"  Revenue: Expected {actual_rev}, Got {extracted_rev} (Err: {rev_err:.4f})")
        print(f"  Net Income: Expected {actual_ni}, Got {extracted_ni} (Err: {ni_err:.4f})")
        print(f"  Debt: Expected {actual_debt}, Got {extracted_debt} (Err: {debt_err:.4f})")
        print(f"  Cash: Expected {actual_cash}, Got {extracted_cash} (Err: {cash_err:.4f})")
        print(f"  Accuracy: {accuracy*100:.1f}%")

        results.append({
            "ticker": ticker,
            "accuracy": accuracy
        })

    # 3. Summary
    avg_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0
    print(f"\n--- Evaluation Complete ---")
    print(f"Average Accuracy: {avg_acc*100:.1f}%")

    if avg_acc > 0.9:
        print("PASS: RAG Pipeline meets quality standards.")
    else:
        print("FAIL: RAG Pipeline needs improvement.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_eval())

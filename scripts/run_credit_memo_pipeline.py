#!/usr/bin/env python3
import sys
import os
import json
import logging
import argparse
from typing import Dict, Any

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.orchestrators.credit_memo_orchestrator import CreditMemoOrchestrator

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RuntimePipeline")

def main():
    parser = argparse.ArgumentParser(description="Run Credit Memo Pipeline for a specific ticker.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., MSFT)")
    parser.add_argument("--sector", type=str, default=None, help="Sector (optional, for synthetic data)")
    parser.add_argument("--output-dir", type=str, default="showcase/data", help="Output directory")
    args = parser.parse_args()

    # Initialize Orchestrator (empty mock library, we rely on synthetic generation)
    orchestrator = CreditMemoOrchestrator(mock_library={}, output_dir=args.output_dir)

    logger.info(f"Generating Credit Memo for {args.ticker}...")

    # Process Entity (triggers synthetic generation if not in library)
    # We pass None for data to force lookup/generation
    result = orchestrator.process_entity(args.ticker, data=None)

    if not result:
        logger.error("Failed to generate credit memo.")
        sys.exit(1)

    memo = result["memo"]
    logs = result["interaction_log"]

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save Memo
    filename = f"credit_memo_{args.ticker}.json"
    output_path = os.path.join(args.output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(memo, f, indent=2)

    logger.info(f"Credit Memo saved to {output_path}")

    # Update Library Index (Append or Update)
    index_path = os.path.join(args.output_dir, "credit_memo_library.json")
    library_index = []
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                library_index = json.load(f)
        except Exception:
            pass

    # Check if entry exists
    existing_entry = next((item for item in library_index if item["id"] == args.ticker), None)

    new_entry = {
        "id": args.ticker,
        "borrower_name": memo['borrower_name'],
        "ticker": args.ticker,
        "sector": memo['borrower_details']['sector'],
        "report_date": memo['report_date'],
        "risk_score": memo['risk_score'],
        "file": filename,
        "summary": f"{memo['borrower_name']} ({memo['borrower_details']['sector']}). Risk Score: {memo['risk_score']}."
    }

    if existing_entry:
        library_index.remove(existing_entry)

    library_index.append(new_entry)

    with open(index_path, 'w') as f:
        json.dump(library_index, f, indent=2)

    logger.info(f"Library index updated.")

    # Update Interaction Logs
    log_path = os.path.join(args.output_dir, "risk_legal_interaction.json")
    interaction_logs = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                interaction_logs = json.load(f)
        except Exception:
            pass

    interaction_logs[args.ticker] = logs
    with open(log_path, 'w') as f:
        json.dump(interaction_logs, f, indent=2)

    logger.info("Pipeline Execution Complete.")

if __name__ == "__main__":
    main()

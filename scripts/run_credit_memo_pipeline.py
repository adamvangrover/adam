#!/usr/bin/env python3
import sys
import os
import json
import logging
from datetime import datetime

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enterprise.credit_memo.orchestrator import CreditMemoOrchestrator
from core.enterprise.credit_memo.audit_logger import audit_logger

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CreditMemoPipeline")

    logger.info("Initializing Enterprise Credit Memo System...")
    orchestrator = CreditMemoOrchestrator()

    borrower = "TechCorp Inc."
    query = "Semiconductor market risk"

    logger.info(f"Generating Credit Memo for {borrower}...")
    memo = orchestrator.generate_credit_memo(borrower, query)

    logger.info("Generation Complete.")

    # 1. Save Memo Output
    showcase_dir = os.path.join("showcase", "data")
    if not os.path.exists(showcase_dir):
        os.makedirs(showcase_dir, exist_ok=True)

    memo_file = os.path.join(showcase_dir, "credit_memo_output.json")
    with open(memo_file, "w") as f:
        f.write(memo.model_dump_json(indent=2))
    logger.info(f"Saved Credit Memo to {memo_file}")

    # 2. Extract Audit Log (Latest Entry)
    # The audit_logger writes to a file in core/data/audit_logs/
    # We will read the latest file and copy it to showcase data
    latest_log_file = audit_logger.log_file
    audit_entries = []

    if os.path.exists(latest_log_file):
        with open(latest_log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Filter for this transaction? Or just take all for demo
                    audit_entries.append(entry)
                except json.JSONDecodeError:
                    continue

    audit_output_file = os.path.join(showcase_dir, "credit_memo_audit_log.json")
    with open(audit_output_file, "w") as f:
        json.dump(audit_entries, f, indent=2)
    logger.info(f"Saved Audit Log to {audit_output_file}")

if __name__ == "__main__":
    main()

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
    logger = logging.getLogger("CreditMemoLibraryGen")

    logger.info("Initializing Enterprise Credit Memo Library Generator...")
    orchestrator = CreditMemoOrchestrator()

    # Portfolio of companies to generate
    companies = [
        {"name": "TechCorp Inc.", "query": "Semiconductor market risk"},
        {"name": "Apple Inc.", "query": "App Store regulation impact"},
        {"name": "Tesla Inc.", "query": "EV margin compression"},
        {"name": "JPMorgan Chase", "query": "Commercial Real Estate exposure"}
    ]

    library = []

    showcase_dir = os.path.join("showcase", "data")
    if not os.path.exists(showcase_dir):
        os.makedirs(showcase_dir, exist_ok=True)

    for company in companies:
        name = company["name"]
        logger.info(f"Generating Credit Memo for {name}...")

        try:
            memo = orchestrator.generate_credit_memo(name, company["query"])

            # Save individual memo
            safe_name = name.replace(" ", "_").replace(".", "")
            filename = f"credit_memo_{safe_name}.json"
            filepath = os.path.join(showcase_dir, filename)

            with open(filepath, "w") as f:
                f.write(memo.model_dump_json(indent=2))

            logger.info(f"Saved {name} to {filename}")

            # Add to library index
            library.append({
                "id": safe_name,
                "borrower_name": memo.borrower_name,
                "report_date": memo.report_date,
                "risk_score": memo.risk_score,
                "file": filename,
                "summary": memo.executive_summary[:150] + "..."
            })

        except Exception as e:
            logger.error(f"Failed to generate for {name}: {e}")

    # Save Library Index
    index_file = os.path.join(showcase_dir, "credit_memo_library.json")
    with open(index_file, "w") as f:
        json.dump(library, f, indent=2)
    logger.info(f"Saved Library Index to {index_file}")

    # 2. Extract Audit Log (Latest Entries)
    # The audit_logger writes to a file in core/data/audit_logs/
    # We will read the latest file and copy it to showcase data
    latest_log_file = audit_logger.log_file
    audit_entries = []

    if os.path.exists(latest_log_file):
        with open(latest_log_file, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    audit_entries.append(entry)
                except json.JSONDecodeError:
                    continue

    audit_output_file = os.path.join(showcase_dir, "credit_memo_audit_log.json")
    with open(audit_output_file, "w") as f:
        json.dump(audit_entries, f, indent=2)
    logger.info(f"Saved Audit Log to {audit_output_file}")

if __name__ == "__main__":
    main()

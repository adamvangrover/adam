#!/usr/bin/env python3
import sys
import os
import json
import logging
import asyncio
from datetime import datetime

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Enterprise Orchestrator (Preferred Architecture)
from core.enterprise.credit_memo.orchestrator import CreditMemoOrchestrator
from core.enterprise.credit_memo.audit_logger import audit_logger

# Import MockEdgar for fallback/setup if needed by the orchestrator (Optional, keeping context)
from core.pipelines.mock_edgar import MockEdgar

# Rich Portfolio Data (From Left Side - Better Dataset)
PORTFOLIO = [
    {"ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "focus": "App Store regulation and hardware margins"},
    {"ticker": "MSFT", "name": "Microsoft Corp", "sector": "Technology", "focus": "Cloud growth and AI integration costs"},
    {"ticker": "NVDA", "name": "NVIDIA Corp", "sector": "Technology", "focus": "Semiconductor demand sustainability"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "focus": "Ad revenue stability and antitrust risks"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer", "focus": "Logistics costs and AWS margins"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "sector": "Consumer", "focus": "EV margin compression and production scaling"},
    {"ticker": "META", "name": "Meta Platforms", "sector": "Technology", "focus": "Ad targeting efficiency and metaverse spend"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "sector": "Financial", "focus": "Commercial Real Estate exposure and net interest margin"},
    {"ticker": "GS", "name": "Goldman Sachs", "sector": "Financial", "focus": "Investment banking deal flow recovery"},
    {"ticker": "BAC", "name": "Bank of America", "sector": "Financial", "focus": "Consumer credit quality and deposit beta"}
]

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("CreditMemoLibraryGen")

    logger.info("Initializing Enterprise Credit Memo Library Generator...")
    
    # Initialize the robust Enterprise Orchestrator
    orchestrator = CreditMemoOrchestrator()

    # Setup directories
    showcase_dir = os.path.join("showcase", "data")
    if not os.path.exists(showcase_dir):
        os.makedirs(showcase_dir, exist_ok=True)

    library_index = []

    # Process the Portfolio
    for entity in PORTFOLIO:
        name = entity["name"]
        ticker = entity["ticker"]
        sector = entity["sector"]
        
        # Construct a query dynamically (Merging Left's data into Right's query logic)
        query = f"{sector} sector risks, {entity['focus']}, and {ticker} financial performance"
        
        logger.info(f"Processing {ticker} - {name}...")

        try:
            # Generate Memo using the Enterprise Orchestrator
            # Note: Assuming orchestrator handles the async/sync internal complexity
            memo = orchestrator.generate_credit_memo(name, query)

            # 1. Save Individual Memo File
            safe_name = name.replace(" ", "_").replace(".", "")
            filename = f"credit_memo_{safe_name}.json"
            filepath = os.path.join(showcase_dir, filename)

            with open(filepath, "w") as f:
                f.write(memo.model_dump_json(indent=2))

            logger.info(f"Saved generated memo to {filename}")

            # 2. Add to Library Index
            # This structure supports the frontend library view
            library_index.append({
                "id": safe_name,
                "borrower_name": memo.borrower_name,
                "ticker": ticker, # Preserved metadata
                "sector": sector, # Preserved metadata
                "report_date": memo.report_date,
                "risk_score": memo.risk_score,
                "file": filename,
                "summary": memo.executive_summary[:150] + "..." if memo.executive_summary else "Summary pending..."
            })

        except Exception as e:
            logger.error(f"Failed to generate for {name}: {e}")
            # Continue to next entity even if one fails

    # 3. Save Library Index (The catalog file)
    index_path = os.path.join(showcase_dir, "credit_memo_library.json")
    with open(index_path, "w") as f:
        json.dump(library_index, f, indent=2)
    logger.info(f"Library generation complete. Index saved to {index_path}")

    # 4. Extract Audit Log (Enterprise Feature)
    # Copies the internal audit log to the showcase data for frontend visualization
    latest_log_file = audit_logger.log_file
    audit_entries = []

    if os.path.exists(latest_log_file):
        try:
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
            logger.info(f"Audit logs exported to {audit_output_file}")
            
        except Exception as e:
            logger.warning(f"Could not process audit logs: {e}")

if __name__ == "__main__":
    main()
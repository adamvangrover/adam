import asyncio
import json
import os
import sys
import logging
from typing import Dict, Any

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.credit.orchestrator import CreditOrchestrator
from core.audit.audit_agent import AuditAgent
from core.audit.audit_logger import AuditLogger

# Configure Logging
logging.basicConfig(level=logging.INFO)

async def verify_credit_backend():
    print("Starting Credit Backend Verification...")

    # 1. Setup Mock Data
    mock_data_path = "verification_credit_mock_data.json"
    mock_data = {
        "Acme Corp": {
            "documents": [
                {
                    "doc_id": "doc_123",
                    "chunks": [
                        {"chunk_id": "chunk_456", "type": "text", "content": "Revenue grew by 15%."},
                        {"chunk_id": "chunk_789", "type": "financial_table", "content_json": {
                            "total_assets": 100.0,
                            "total_liabilities": 40.0,
                            "total_equity": 60.0, # Balanced
                            "ebitda": 20.0,
                            "total_debt": 50.0
                        }}
                    ]
                }
            ],
            "market_data": "Bullish outlook on widget sector."
        }
    }

    with open(mock_data_path, 'w') as f:
        json.dump(mock_data, f)

    try:
        # 2. Instantiate Orchestrator
        config = {
            "mock_data_path": mock_data_path,
            "agent_id": "credit_orchestrator_test"
        }
        orchestrator = CreditOrchestrator(config)

        # 3. Execute Workflow
        request = {"borrower_name": "Acme Corp"}
        result = await orchestrator.generate_credit_memo(request)

        print(f"Memo Generated: {result.get('memo', {}).get('output')[:50]}...")

        # 4. Verify Audit Log
        logger = AuditLogger() # defaults to data/audit_log.jsonl
        logs = logger.get_logs()

        if not logs:
            print("FAIL: No audit logs found.")
        else:
            latest_log = logs[-1]
            print(f"PASS: Audit log found for transaction {latest_log.transaction_id}")

            # 5. Run Audit Agent Checks
            # Check prompt reproducibility
            # We mock the registry for this check since we haven't implemented a full git-based one
            registry_mock = {latest_log.prompt_version_id: True}
            audit_res = AuditAgent.audit_prompt_reproducibility(latest_log.dict(), registry_mock)
            print(f"Audit (Prompt): {audit_res}")

            # Check citation density
            memo_text = result['memo']['output']
            audit_cit = AuditAgent.audit_citation_density(memo_text)
            print(f"Audit (Citations): {audit_cit}")

            if "FAIL" in audit_res or "FAIL" in audit_cit:
                print("FAIL: Audit checks failed.")
            else:
                print("PASS: Backend Verification Complete.")

    except Exception as e:
        print(f"FAIL: Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(mock_data_path):
            os.remove(mock_data_path)

if __name__ == "__main__":
    asyncio.run(verify_credit_backend())

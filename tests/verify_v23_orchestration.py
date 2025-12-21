from core.engine.states import init_risk_state
from core.xai.state_translator import ExplainableStateTranslator
from core.engine.meta_orchestrator import MetaOrchestrator
import sys
import os
import logging
import asyncio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


def verify_orchestration():
    print("\n=== Meta-Orchestrator Verification ===")
    orchestrator = MetaOrchestrator()

    queries = [
        "Get stock price of AAPL",  # Low
        "Monitor AAPL news",  # Medium
        "Analyze Apple Inc. Credit Risk"  # High (v23)
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        result = asyncio.run(orchestrator.route_request(q))
        print(f"Result: {result}")

        # If v23, we can inspect the result
        if isinstance(result, dict) and "final_state" in result:
            final_state = result["final_state"]
            xai_msg = ExplainableStateTranslator.generate_user_update(final_state)
            print(f"XAI Update: {xai_msg}")


def main():
    verify_orchestration()


if __name__ == "__main__":
    main()

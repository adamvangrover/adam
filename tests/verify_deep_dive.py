import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)

from core.engine.deep_dive_graph import deep_dive_app
from core.engine.states import init_omniscient_state


async def main():
    print("Starting Deep Dive Graph Verification...")

    initial_state = init_omniscient_state("AAPL")
    config = {"configurable": {"thread_id": "test_dd_1"}}

    print(f"Initial State: {initial_state['human_readable_status']}")

    try:
        # Use ainvoke if async, but deep_dive_graph is synchronous nodes wrapped in StateGraph.
        # LangGraph invoke can handle it.
        if hasattr(deep_dive_app, 'ainvoke'):
            result = await deep_dive_app.ainvoke(initial_state, config=config)
        else:
            result = deep_dive_app.invoke(initial_state, config=config)

        print("\n--- Final Result ---")
        print(f"Final Status: {result.get('human_readable_status')}")
        kg = result.get('v23_knowledge_graph', {}).get('nodes', {})

        print("\n--- Knowledge Graph Nodes ---")
        print(f"Entity: {kg.get('entity_ecosystem', {}).get('legal_entity', {}).get('name')}")
        print(f"Valuation: {kg.get('equity_analysis', {}).get('valuation_engine', {}).get('dcf_model', {}).get('intrinsic_share_price')}")
        print(f"Credit: {kg.get('credit_analysis', {}).get('snc_rating_model', {}).get('overall_borrower_rating')}")
        print(f"Recommendation: {kg.get('strategic_synthesis', {}).get('final_verdict', {}).get('recommendation')}")

    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.v23_graph_engine.cyclical_reasoning_graph import cyclical_reasoning_app
from core.v23_graph_engine.states import init_risk_state

def setup_dummy_data():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/risk_rating_mapping.json'):
        with open('data/risk_rating_mapping.json', 'w') as f:
            json.dump({"metadata": {"debug_mode": True}, "risk_weights": {}}, f)
    if not os.path.exists('data/adam_market_baseline.json'):
        with open('data/adam_market_baseline.json', 'w') as f:
            json.dump({"market_index": "S&P 500", "price_data": [100, 101]}, f)

def main():
    print("Starting v23 Graph Verification...")
    setup_dummy_data()

    # Use ABC_TEST to get valid mock data from DataRetrievalAgent
    initial_state = init_risk_state("ABC_TEST", "Assess credit risk")
    config = {"configurable": {"thread_id": "verification_run_2"}}

    print(f"Initial State: {initial_state['human_readable_status']}")

    # Run the graph
    print("\n--- Execution Trace ---")
    try:
        result = cyclical_reasoning_app.invoke(initial_state, config=config)

        print("\n--- Final Result ---")
        print(f"Final Status: {result.get('human_readable_status')}")
        print(f"Iteration Count: {result.get('iteration_count')}")
        print(f"Quality Score: {result.get('quality_score')}")

        if "draft_analysis" in result:
            print("Draft Analysis Preview:\n")
            print(result["draft_analysis"])

    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

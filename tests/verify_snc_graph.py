# tests/verify_snc_graph.py

from core.engine.snc_graph import snc_graph_app
from core.engine.states import init_snc_state
import json


def test_snc_graph():
    print("Running SNC Graph Verification...")

    # 1. Define Input Data
    obligor_id = "Titan Energy Partners"
    syndicate = {
        "banks": [
            {"name": "BigBank", "role": "Lead", "share": 0.6},
            {"name": "SmallBank", "role": "Participant", "share": 0.1},
            {"name": "MedBank", "role": "Participant", "share": 0.3}
        ]
    }
    financials = {
        "ebitda": 350.0,
        "total_debt": 1800.0,
        "liquidity": 150.0
    }

    # 2. Initialize State
    initial_state = init_snc_state(obligor_id, syndicate, financials)

    # 3. Run Graph
    # langgraph invocation
    final_state = snc_graph_app.invoke(initial_state, config={"configurable": {"thread_id": "verify_1"}})

    print(f"Final Rating: {final_state['regulatory_rating']}")
    print(f"Iterations: {final_state['iteration_count']}")
    print("-" * 20)
    print(final_state['rationale'])

    # Assertions
    assert final_state['regulatory_rating'] in ["Pass", "Special Mention", "Substandard"]
    assert final_state['iteration_count'] >= 1
    print("\nVerification Passed!")


if __name__ == "__main__":
    test_snc_graph()

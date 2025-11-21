import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.v23_graph_engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.v23_graph_engine.autonomous_self_improvement import AutonomousSelfImprovementController
from core.v23_graph_engine.states import init_risk_state

# Configure logging to see the flow
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_planner():
    print("\n=== Phase 2: Neuro-Symbolic Planner Verification ===")
    planner = NeuroSymbolicPlanner()

    query = "Analyze Apple Inc. Credit Rating"
    print(f"Query: {query}")

    # 1. Discover Plan
    plan = planner.discover_plan(query)
    if plan:
        print("\nSymbolic Plan Discovered:")
        for step in plan:
            print(f"  ({step['source']}) --[{step['relation']}]--> ({step['target']}) [Prov: {step['provenance']}]")

        # 2. Compile to Graph
        app = planner.to_executable_graph(plan)
        if app:
            print("\nExecuting Compiled Graph:")
            state = init_risk_state("AAPL", query)
            try:
                # The generated graph nodes accept 'state' but might ignore it in this scaffolding
                result = app.invoke(state)
                print("Execution Complete.")
            except Exception as e:
                print(f"Execution Error: {e}")
    else:
        print("Failed to discover plan.")

def verify_self_improvement():
    print("\n=== Phase 3: Self-Improvement Verification ===")
    controller = AutonomousSelfImprovementController()

    agent_name = "RiskAssessmentAgent"
    print(f"Monitoring {agent_name}...")

    # Simulate failures
    failures = [
        "Timeout waiting for market data",
        "Invalid JSON output",
        "Hallucinated financial ratio"
    ]

    for i, error in enumerate(failures):
        print(f"Simulating failure {i+1}: {error}")
        controller.log_failure(agent_name, "Analyze AAPL", error)

    # The controller should trigger the loop on the 3rd failure
    print("Check logs above for 'ADAPTATION LOOP' messages.")

def main():
    verify_planner()
    verify_self_improvement()

if __name__ == "__main__":
    main()

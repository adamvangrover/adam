import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.neuro_symbolic_planner import NeuroSymbolicPlanner
from core.engine.autonomous_self_improvement import AutonomousSelfImprovementController
from core.engine.cyclical_reasoning_graph import cyclical_reasoning_app
from core.engine.states import init_risk_state

# Configure logging to see the flow
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_planner():
    print("\n=== Phase 2: Neuro-Symbolic Planner Verification ===")
    planner = NeuroSymbolicPlanner()
    
    query = "Analyze Apple Inc. Credit Rating"
    print(f"Query: {query}")
    
    # 1. Discover Plan
    # Updated to match actual method signature: discover_plan(start_node, target_node)
    result = planner.discover_plan(start_node="Apple Inc.", target_node="Credit Default")
    plan = result.get("raw_path", [])
    if plan:
        print("\nSymbolic Plan Discovered:")
        for step in plan:
            print(f"  ({step['source']}) --[{step['relation']}]--> ({step['target']})")
            
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

def verify_cyclical_graph():
    print("\n=== Phase 1: Cyclical Reasoning Graph Verification ===")
    state = init_risk_state("TSLA", "Full Risk Analysis")

    # Needs a config for memory (thread_id)
    config = {"configurable": {"thread_id": "verify_v23"}}

    print(f"Invoking Cyclical Graph for {state['ticker']}...")
    try:
        final_state = cyclical_reasoning_app.invoke(state, config=config)

        print("\nFinal State Summary:")
        print(f"Status: {final_state['human_readable_status']}")
        print(f"Iterations: {final_state['iteration_count']}")
        print(f"Quality Score: {final_state['quality_score']}")
        print(f"Needs Correction: {final_state['needs_correction']}")

        if final_state['quality_score'] > 0.8:
            print("SUCCESS: Graph converged to high quality.")
        else:
            print("WARNING: Graph did not converge.")

    except Exception as e:
        print(f"Graph Execution Failed: {e}")

def main():
    verify_cyclical_graph()
    verify_planner()
    verify_self_improvement()

if __name__ == "__main__":
    main()

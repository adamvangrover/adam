"""
Verification Script for Simulation Logic and Provenance Logging.
"""

import sys
import os
import networkx as nx
import json

# Add root to path
sys.path.append(os.getcwd())

from core.v23_graph_engine.simulation_engine import CrisisSimulationEngine
from core.system.provenance_logger import ProvenanceLogger

def test_simulation_workflow():
    print(">>> 1. Setting up Environment")
    # Setup Mock Graph
    kg = nx.DiGraph()
    kg.add_node("TSM", type="Equity", valuation=100.0, sector="Semiconductors")
    kg.add_node("AAPL", type="Equity", valuation=180.0, sector="Tech")
    kg.add_node("NDX", type="Index", valuation=15000.0)

    # Edges: TSM -> AAPL (Supplier dependency)
    kg.add_edge("TSM", "AAPL", weight=0.9, type="supply_chain")
    kg.add_edge("TSM", "NDX", weight=0.5, type="constituent")

    print(">>> 2. Initializing Engine and Logger")
    logger = ProvenanceLogger()
    engine = CrisisSimulationEngine(kg, logger=logger)

    # Load the prompt we created earlier
    prompt_path = "prompt_library/AOPL-v1.0/simulation/semiconductor_supply_shock.md"
    if not os.path.exists(prompt_path):
        print(f"FAILED: Prompt file {prompt_path} not found.")
        return

    print(f">>> 3. Loading Scenario from {prompt_path}")
    scenario = engine.load_scenario_from_markdown(prompt_path)
    print(f"    Loaded Scenario: {scenario['title']} (ID: {scenario['id']})")
    print(f"    Shocks detected: {len(scenario['shocks'])}")
    for shock in scenario['shocks']:
         print(f"      - {shock['target']}: {shock['change']}")

    if not scenario['shocks']:
         print("    WARNING: No shocks parsed. Check regex.")

    print(">>> 4. Running Simulation")
    try:
        result = engine.run_simulation(scenario['id'])
        print("    Simulation Success.")
        print("    Direct Impacts:")
        for impact in result['direct_impacts']:
            print(f"      - {impact['node']}: {impact['shock']*100}%")
    except Exception as e:
        print(f"FAILED: Simulation execution error: {e}")
        import traceback
        traceback.print_exc()
        return

    print(">>> 5. Verifying Provenance Log")
    history = logger.get_session_history()

    found_sim_log = False
    for entry in history:
        if entry['activity_type'] == "simulation":
            found_sim_log = True
            print(f"    Found Log: {entry['record_id']} @ {entry['timestamp']}")
            print(f"    Log content sample: {str(entry['outcome'])[:100]}...")
            break

    if found_sim_log:
        print("PASS: Simulation logic and logging verified.")
    else:
        print("FAIL: No simulation log found.")

if __name__ == "__main__":
    test_simulation_workflow()

import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation.generator import SimulationGenerator

def generate_nexus_data():
    print("Initializing Nexus Simulation...")
    # Increase scale for "Board Level" complexity
    sim = SimulationGenerator(num_sovereigns=20)

    print("Running Simulation Steps...")
    results = sim.generate_full_simulation(steps=50)

    # 1. Prepare Frontend Data (Graph format)
    nodes = []
    edges = []

    for sov in results["sovereigns"]:
        # Node
        nodes.append({
            "id": sov["id"],
            "label": sov["name"],
            "group": sov["ideology"].split(" ")[-1].lower(), # e.g. "autocracy", "democracy"
            "value": sov["stability_index"] * 20, # Size based on stability
            "title": f"Ideology: {sov['ideology']}<br>GDP Growth: {sov['gdp_growth']}<br>Stability: {sov['stability_index']:.2f}"
        })

        # Edges (Allies/Adversaries)
        # We need to map names back to IDs for edges
        name_to_id = {s["name"]: s["id"] for s in results["sovereigns"]}

        for ally_name in sov["allies"]:
            if ally_name in name_to_id:
                edges.append({
                    "from": sov["id"],
                    "to": name_to_id[ally_name],
                    "color": {"color": "#10b981"}, # Green for ally
                    "dashes": False
                })

        for enemy_name in sov["adversaries"]:
            if enemy_name in name_to_id:
                edges.append({
                    "from": sov["id"],
                    "to": name_to_id[enemy_name],
                    "color": {"color": "#ef4444"}, # Red for enemy
                    "dashes": True
                })

    frontend_data = {
        "metadata": {
            "title": "NEXUS OF EXPOSURE: SIMULATION RUN 001",
            "version": "1.0",
            "description": "High-fidelity geopolitical simulation with synthetic cognitive streams."
        },
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "metrics": {
            "global_stability": sum(s["stability_index"] for s in results["sovereigns"]) / len(results["sovereigns"]),
            "conflict_zones": len([e for e in results["events"] if e["type"] == "Civil Unrest"]),
            "timeline": results["events"]
        },
        "stream": results["synthetic_data"][-100:] # Last 100 thoughts for the UI stream
    }

    # Output Paths
    showcase_data_path = "showcase/data/nexus_simulation.json"
    training_data_path = "data/synthetic_training/sovereign_minds.jsonl"

    # Write Frontend JSON
    print(f"Writing frontend data to {showcase_data_path}...")
    with open(showcase_data_path, 'w') as f:
        json.dump(frontend_data, f, indent=2)

    # Write Synthetic Training Data (JSONL)
    print(f"Writing synthetic training data to {training_data_path}...")
    with open(training_data_path, 'w') as f:
        for thought in results["synthetic_data"]:
            f.write(json.dumps({"text": thought}) + "\n")

    print("Simulation Generation Complete.")

if __name__ == "__main__":
    generate_nexus_data()

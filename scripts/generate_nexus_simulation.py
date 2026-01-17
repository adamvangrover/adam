import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulation.generator import SimulationGenerator

def generate_nexus_data():
    print("Initializing Nexus Simulation (Board Level Depth)...")
    # Increase scale for "Board Level" complexity
    sim = SimulationGenerator(num_sovereigns=50)

    print("Running Simulation Steps (Time Horizon: 100)...")
    results = sim.generate_full_simulation(steps=100)

    # 1. Prepare Frontend Data (Graph format)
    nodes = []
    edges = []

    name_to_id = {s["name"]: s["id"] for s in results["sovereigns"]}

    for sov in results["sovereigns"]:
        # Extract rich metrics
        econ = sov["economy"]
        mil = sov["military"]

        # Tooltip content
        tooltip = (
            f"<b>{sov['name']}</b><br>"
            f"Ideology: {sov['ideology']}<br>"
            f"Stability: {sov['stability_index']:.2f}<br>"
            f"GDP Growth: {econ['gdp_growth']*100:.1f}%<br>"
            f"Inflation: {econ['inflation_rate']*100:.1f}%<br>"
            f"Mil. Readiness: {mil['readiness']*100:.0f}%"
        )

        # Node
        nodes.append({
            "id": sov["id"],
            "label": sov["name"],
            "group": sov["ideology"].split(" ")[-1].lower(), # e.g. "autocracy", "democracy"
            "value": sov["stability_index"] * 20 + (econ["gdp_growth"] * 100), # Size based on stability + growth
            "title": tooltip
        })

        # Edges (Allies/Adversaries)
        for ally_name in sov["allies"]:
            if ally_name in name_to_id:
                edges.append({
                    "from": sov["id"],
                    "to": name_to_id[ally_name],
                    "color": {"color": "#10b981", "opacity": 0.4}, # Green for ally
                    "dashes": False,
                    "width": 1
                })

        for enemy_name in sov["adversaries"]:
            if enemy_name in name_to_id:
                edges.append({
                    "from": sov["id"],
                    "to": name_to_id[enemy_name],
                    "color": {"color": "#ef4444", "opacity": 0.4}, # Red for enemy
                    "dashes": True,
                    "width": 1
                })

    frontend_data = {
        "metadata": {
            "title": "NEXUS OF EXPOSURE: GLOBAL SIMULATION RUN 002",
            "version": "2.0",
            "description": "High-fidelity geopolitical simulation with economic, demographic, and military vectors."
        },
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "metrics": {
            "global_stability": sum(s["stability_index"] for s in results["sovereigns"]) / len(results["sovereigns"]),
            "avg_inflation": sum(s["economy"]["inflation_rate"] for s in results["sovereigns"]) / len(results["sovereigns"]),
            "conflict_events": len([e for e in results["events"] if "Civil Unrest" in e["type"] or "Military" in e["type"]]),
            "timeline": results["events"]
        },
        "stream": results["synthetic_data"][-150:] # Last 150 thoughts for the UI stream
    }

    # Output Paths
    showcase_data_path = "showcase/data/nexus_simulation.json"
    training_data_path = "data/synthetic_training/sovereign_minds.jsonl"

    os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(showcase_data_path), exist_ok=True)

    # Write Frontend JSON
    print(f"Writing frontend data to {showcase_data_path}...")
    with open(showcase_data_path, 'w') as f:
        json.dump(frontend_data, f, indent=2)

    # Write Synthetic Training Data (JSONL)
    print(f"Writing synthetic training data to {training_data_path}...")
    with open(training_data_path, 'w') as f:
        for thought in results["synthetic_data"]:
            f.write(json.dumps({"text": thought}) + "\n")

    print(f"Simulation Generation Complete. Generated {len(nodes)} sovereigns and {len(results['events'])} events.")

if __name__ == "__main__":
    generate_nexus_data()

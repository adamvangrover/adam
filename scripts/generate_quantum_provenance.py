import json
import time
import random
import os
import sys
from datetime import datetime, timedelta

# Ensure python path includes repo root for imports
sys.path.append(os.getcwd())

try:
    from core.simulations.align_future_simulator import AlignFutureSimulator
except ImportError as e:
    print(f"Warning: Could not import AlignFutureSimulator. Dependencies might be missing: {e}")
    AlignFutureSimulator = None

def generate_quantum_provenance():
    print("Initializing Quantum/Enterprise Simulation Data Generator...")

    # 1. Generate Provenance Logs
    modules = ["QuantumSearch", "DataLake_Ingest", "Vector_Index", "CrossCloud_Sync", "Heuristic_Annealer"]
    operations = ["optimize", "sync", "index", "shred", "entangle", "collapse"]
    statuses = ["SUCCESS", "SUCCESS", "SUCCESS", "WARNING", "OPTIMIZED"]

    logs = []
    base_time = datetime.now()

    for i in range(50):
        dt = base_time - timedelta(minutes=random.randint(0, 60))
        module = random.choice(modules)
        log = {
            "id": f"evt_{random.randint(10000,99999)}",
            "timestamp": dt.timestamp(),
            "iso_time": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "module": module,
            "operation": random.choice(operations),
            "status": random.choice(statuses),
            "inputs": {"qubits": random.randint(128, 1024), "shards": random.randint(1, 10)},
            "hash": f"sha256:{random.getrandbits(256):064x}"[:16] + "..."
        }
        logs.append(log)

    # Sort logs by time (newest first)
    logs.sort(key=lambda x: x['timestamp'], reverse=True)

    # 2. Generate Infrastructure Topology
    nodes = [
        {"id": "AWS_US_EAST", "label": "AWS Data Lake", "type": "cloud", "status": "active"},
        {"id": "GCP_EUROPE", "label": "GCP Analytics", "type": "cloud", "status": "active"},
        {"id": "AZURE_ASIA", "label": "Azure AI Core", "type": "cloud", "status": "active"},
        {"id": "ON_PREM_CORE", "label": "Quantum Core (On-Prem)", "type": "quantum", "status": "superposition"},
        {"id": "EDGE_FLEET", "label": "Edge Nodes", "type": "edge", "status": "idle"}
    ]

    links = [
        {"source": "ON_PREM_CORE", "target": "AWS_US_EAST", "throughput": "100 Gbps"},
        {"source": "ON_PREM_CORE", "target": "GCP_EUROPE", "throughput": "100 Gbps"},
        {"source": "AWS_US_EAST", "target": "AZURE_ASIA", "throughput": "50 Gbps"},
        {"source": "GCP_EUROPE", "target": "EDGE_FLEET", "throughput": "10 Gbps"}
    ]

    # 3. Run Align Future Simulation
    simulation_data = {}
    if AlignFutureSimulator:
        print("Running Align Future Simulator...")
        try:
            sim = AlignFutureSimulator(".")
            sim.scan_repository()
            # Generate 20 steps of evolution
            timeline = sim.evolve_state(steps=20)

            # Extract static qubit info for the lattice visualization
            qubit_registry = [
                {"name": q.name, "type": q.type_label, "initial_complexity": q.complexity}
                for q in sim.qubits.values()
            ]

            simulation_data = {
                "available": True,
                "qubit_registry": qubit_registry,
                "timeline": timeline,
                "qubit_count": len(qubit_registry)
            }
            print(f"Simulation complete. Generated {len(timeline)} steps for {len(qubit_registry)} qubits.")
        except Exception as e:
            print(f"Simulation failed: {e}")
            simulation_data = {"available": False, "error": str(e)}
    else:
        simulation_data = {"available": False, "error": "Import failed"}

    # 4. Compile Output
    data = {
        "logs": logs,
        "infrastructure": {
            "nodes": nodes,
            "links": links
        },
        "metrics": {
            "active_qubits": simulation_data.get("qubit_count", 1024),
            "coherence_time": "12.4ms",
            "global_latency": "42ms",
            "search_space_coverage": "99.9%"
        },
        "simulation": simulation_data,
        "generated_at": datetime.now().isoformat()
    }

    # 5. Write to JS File
    output_path = "showcase/js/mock_quantum_data.js"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    js_content = f"window.QUANTUM_DATA = {json.dumps(data, indent=2)};"

    with open(output_path, "w") as f:
        f.write(js_content)

    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    generate_quantum_provenance()

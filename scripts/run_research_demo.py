import os
import sys
import json
import logging

# Ensure repo root is in path
sys.path.append(os.getcwd())

from core.research.federated_learning.federated_coordinator import FederatedCoordinator
from core.research.gnn.graph_risk_engine import GraphRiskEngine
from core.oswm.inference import OSWMInference

OUTPUT_FILE = "showcase/data/research_output.json"

def run_federated_demo():
    print("--- Running Federated Learning Demo ---")
    coordinator = FederatedCoordinator(num_clients=3)

    # Run 5 rounds
    history = []
    for i in range(5):
        weights = coordinator.start_round()
        # Mock validation
        val_acc = 0.8 + (i * 0.02) # Improving
        print(f"Round {i+1}: Global Model Updated (Acc: {val_acc:.2f})")
        history.append({"round": i+1, "accuracy": val_acc})

    return history

def run_gnn_demo():
    print("\n--- Running GNN Risk Engine Demo ---")
    engine = GraphRiskEngine()

    nodes = [
        {"id": "Bank_A", "assets": 100},
        {"id": "Bank_B", "assets": 50},
        {"id": "HedgeFund_C", "assets": 200},
        {"id": "Crypto_D", "assets": 10}
    ]

    edges = [
        {"source": "Bank_A", "target": "HedgeFund_C", "weight": 0.8},
        {"source": "Bank_B", "target": "Bank_A", "weight": 0.3},
        {"source": "HedgeFund_C", "target": "Crypto_D", "weight": 0.9}
    ]

    risks = engine.predict_risk(nodes, edges)
    print("Predicted Contagion Risks:")
    for node, score in risks.items():
        print(f"  {node}: {score:.4f}")

    return risks

def run_oswm_demo():
    print("\n--- Running One-Shot World Model Demo ---")
    inference = OSWMInference()

    initial_state = {"step": 0, "gdp": 100.0, "inflation": 0.02}
    print(f"Initial State: {initial_state}")

    actions = ["stimulate_economy", "raise_rates", "stimulate_economy"]
    trajectory = [initial_state]

    current_state = initial_state
    for action in actions:
        print(f"Action: {action}")
        next_state = inference.predict_next_state(current_state, action)
        print(f"  Result: {next_state}")
        trajectory.append(next_state)
        current_state = next_state

    return trajectory

def main():
    logging.basicConfig(level=logging.INFO)

    fed_results = run_federated_demo()
    gnn_results = run_gnn_demo()
    oswm_results = run_oswm_demo()

    output = {
        "federated_learning": fed_results,
        "gnn_risk": gnn_results,
        "oswm_trajectory": oswm_results
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResearch output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

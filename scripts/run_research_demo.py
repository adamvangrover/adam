import sys
import os
import json
import logging
import random
import torch

# Ensure core is in path
sys.path.append(os.getcwd())

from core.research.gnn.engine import GraphRiskEngine
from core.research.federated_learning.fl_coordinator import FederatedCoordinator
from core.research.oswm.inference import OSWMInference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResearchDemo")

def run_gnn_analysis():
    logger.info("--- Starting Graph Neural Network Risk Analysis ---")
    try:
        engine = GraphRiskEngine()
        logger.info(f"Graph Engine initialized with {engine.num_nodes} nodes.")

        risk_scores = engine.predict_risk()

        # Top 5 riskiest nodes
        sorted_risk = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        top_risky = sorted_risk[:5]

        logger.info("Top 5 High-Risk Nodes detected by GNN:")
        for node, score in top_risky:
            logger.info(f"  {node}: {score:.4f}")

        return sorted_risk
    except Exception as e:
        logger.error(f"GNN Analysis failed: {e}")
        return []

def run_federated_learning():
    logger.info("--- Starting Federated Learning Simulation ---")
    coordinator = FederatedCoordinator(num_clients=3, input_dim=10)

    history = []
    for round_num in range(1, 6):
        loss, acc = coordinator.run_round(round_num)
        history.append({"round": round_num, "loss": loss, "accuracy": acc})

    logger.info("FL Simulation Complete.")
    return history

def run_oswm_simulation():
    logger.info("--- Starting One-Shot World Model Simulation ---")
    oswm = OSWMInference()

    logger.info("Pre-training on synthetic prior (Physics/Sine waves)...")
    oswm.pretrain_on_synthetic_prior(steps=50)

    # Create a synthetic "Crisis" context (market dropping)
    # Price dropping from 100 to 90 over 10 ticks
    context = [100.0 - i * 1.0 + random.random() for i in range(10)]
    logger.info(f"Context (Market Crash Start): {context}")

    # Predict recovery?
    logger.info("Generating future scenario...")
    future_prices = oswm.generate_scenario(context, steps=10)
    logger.info(f"Predicted Future: {future_prices}")

    return {"context": context, "prediction": future_prices}

def main():
    logger.info("Initializing Research Modules...")

    # 1. GNN
    gnn_results = run_gnn_analysis()

    # 2. FL
    fl_history = run_federated_learning()

    # 3. OSWM
    oswm_scenario = run_oswm_simulation()

    # Output structure
    output_data = {
        "gnn_risk_analysis": {
            "top_risky_nodes": gnn_results[:10]
        },
        "federated_learning": {
            "training_history": fl_history
        },
        "oswm_simulation": {
            "scenario_type": "Market Recovery",
            "data": oswm_scenario
        }
    }

    # Save to file
    os.makedirs("showcase/data", exist_ok=True)
    with open("showcase/data/research_output.json", "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Research demo complete. Data saved to showcase/data/research_output.json")

if __name__ == "__main__":
    main()

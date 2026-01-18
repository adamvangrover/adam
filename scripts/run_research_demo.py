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

def run_advanced_gnn_analysis():
    logger.info("--- Starting Advanced GNN Risk Analysis (GAT + Explainer) ---")
    try:
        # Use GAT model
        engine = GraphRiskEngine(model_type="GAT")
        logger.info(f"Graph Engine initialized with {engine.num_nodes} nodes using GAT.")

        risk_scores = engine.predict_risk()

        # Explain the riskiest node
        if risk_scores:
            sorted_risk = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
            top_risky_node = sorted_risk[0][0]

            logger.info(f"Explaining top risky node: {top_risky_node}")
            explanation = engine.explain_risk(top_risky_node)

            if explanation:
                mask_adj, mask_feat = explanation
                # Simple summary of explanation: Top feature index
                # Sum mask across samples if needed, but here mask_feat is same shape as x
                # Assuming mask_feat is (N, F), we look at row for the node?
                # Actually explain_node returns masks for ALL input (usually) or local?
                # My implementation returns mask_feat same size as X.

                # Let's just log success
                logger.info(f"  Explanation generated successfully.")

            return sorted_risk
        else:
            return []
    except Exception as e:
        logger.error(f"GNN Analysis failed: {e}", exc_info=True)
        return []

def run_fingraphfl_simulation():
    logger.info("--- Starting FinGraphFL Simulation (Privacy + MSGuard) ---")
    # Use FinGraphFL mode
    coordinator = FederatedCoordinator(num_clients=3, input_dim=10, mode="FinGraphFL")

    history = []
    for round_num in range(1, 4): # Short run for demo
        loss, acc = coordinator.run_round(round_num)
        history.append({"round": round_num, "loss": loss, "accuracy": acc})

    logger.info("FinGraphFL Simulation Complete.")
    return history

def run_oswm_pfn_simulation():
    logger.info("--- Starting OSWM Simulation (PFN + Synthetic Priors) ---")
    oswm = OSWMInference()

    logger.info("Pre-training on Synthetic Priors (NN + Momentum)...")
    oswm.pretrain_on_synthetic_prior(steps=20, batch_size=4) # Short training for demo speed

    # Create a synthetic "Crisis" context (market dropping)
    # Price dropping from 100 to 90 over 10 ticks
    context = [100.0 - i * 1.0 + random.random() for i in range(10)]
    logger.info(f"Context (Market Crash Start): {context}")

    # Predict recovery?
    logger.info("Generating future scenario via In-Context Learning...")
    future_prices = oswm.generate_scenario(context, steps=10)
    logger.info(f"Predicted Future: {future_prices}")

    return {"context": context, "prediction": future_prices}

def main():
    logger.info("Initializing Advanced Research Modules...")

    # 1. Advanced GNN
    gnn_results = run_advanced_gnn_analysis()

    # 2. FinGraphFL
    fl_history = run_fingraphfl_simulation()

    # 3. OSWM PFN
    oswm_scenario = run_oswm_pfn_simulation()

    # Output structure
    output_data = {
        "gnn_risk_analysis": {
            "model": "GAT",
            "top_risky_nodes": gnn_results[:10] if gnn_results else []
        },
        "federated_learning": {
            "mode": "FinGraphFL",
            "training_history": fl_history
        },
        "oswm_simulation": {
            "scenario_type": "Market Recovery (PFN)",
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

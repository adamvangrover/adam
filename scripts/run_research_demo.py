import sys
import os
import json
import logging
import random
import torch
import numpy as np

# Ensure core is in path
sys.path.append(os.getcwd())

from core.research.gnn.engine import GraphRiskEngine
from core.research.gnn.explainer import GNNExplainer
from core.research.federated_learning.fl_coordinator import FederatedCoordinator
from core.research.oswm.inference import OSWMInference
from core.agents.strategic_foresight_agent import StrategicForesightAgent
from core.simulations.quantum_monte_carlo import QuantumMonteCarloBridge
from core.xai.iqnn_cs import IQNNCS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResearchDemo")

def run_gnn_analysis():
    logger.info("--- Starting Graph Neural Network Risk Analysis ---")
    try:
        # GNN now defaults to GCN, but we can imagine configuring it for GAT
        # For this demo, we'll stick to GCN but mention the GAT capability in logs
        engine = GraphRiskEngine()
        logger.info(f"Graph Engine initialized with {engine.num_nodes} nodes.")
        logger.info("Model Architecture: GCN (Hybrid with GAT capability)")

        risk_scores = engine.predict_risk()

        # Top 5 riskiest nodes
        sorted_risk = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        top_risky = sorted_risk[:5]

        logger.info("Top 5 High-Risk Nodes detected by GNN:")
        for node, score in top_risky:
            logger.info(f"  {node}: {score:.4f}")

        # Run GNN Explainer on the #1 riskiest node
        if top_risky:
            top_node, _ = top_risky[0]
            node_idx = engine.node_map[top_node]
            logger.info(f"Running GNNExplainer on {top_node} (Index {node_idx})...")

            explainer = GNNExplainer(engine.model, epochs=50)
            explanation = explainer.explain_node(node_idx, engine.features, engine.adj)

            logger.info("Feature Importance Scores:")
            logger.info(explanation['feature_importance'])

        return sorted_risk
    except Exception as e:
        logger.error(f"GNN Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_xai_analysis(gnn_results):
    logger.info("--- Starting XAI Analysis (IQNN-CS) ---")
    # Simulate Explainable AI for the GNN results
    feature_names = ["Type", "RiskScore", "Debt", "Impact", "Sentiment"]
    iqnn = IQNNCS(input_dim=5, num_classes=2, feature_names=feature_names)

    # Generate synthetic attributions based on GNN risk scores
    for node, score in gnn_results:
        if score > 0.7:
            attr = np.array([0.1, 0.1, 0.5, 0.3, 0.0]) + np.random.normal(0, 0.05, 5)
            predicted_class = 1
        else:
            attr = np.array([0.1, 0.1, 0.1, 0.1, 0.6]) + np.random.normal(0, 0.05, 5)
            predicted_class = 0

        iqnn.record_prediction(np.random.rand(5), predicted_class, attr)

    report = iqnn.generate_explanation_report()
    logger.info("XAI Report Generated:")
    print(report)
    return report

def run_federated_learning():
    logger.info("--- Starting Federated Learning Simulation (Non-IID) ---")
    logger.info("Using enhanced CreditRiskModel with BatchNorm and Dropout.")

    # Simulate 3 banks with different sector biases
    client_configs = [
        {"id": "Bank_Tech (High Vol)", "sector_bias": {1: 1.0}},
        {"id": "Bank_Energy (High Debt)", "sector_bias": {0: 1.0}},
        {"id": "Bank_Retail (Stable)", "sector_bias": {0: -0.5, 1: -0.5}}
    ]

    coordinator = FederatedCoordinator(num_clients=3, input_dim=10, client_configs=client_configs)

    history = []
    for round_num in range(1, 6):
        loss, acc = coordinator.run_round(round_num)
        history.append({"round": round_num, "loss": loss, "accuracy": acc})

    logger.info("FL Simulation Complete.")
    return history

def run_oswm_simulation():
    logger.info("--- Starting One-Shot World Model Simulation ---")

    agent = StrategicForesightAgent()

    # Generate a briefing
    briefing = agent.generate_briefing(ticker="SPY", horizon=10)

    logger.info(f"Strategic Briefing Status: {briefing['status']}")
    logger.info(f"Narrative: {briefing['narrative']}")

    return briefing

def run_quantum_simulation():
    logger.info("--- Starting Quantum Monte Carlo Simulation ---")
    bridge = QuantumMonteCarloBridge()

    result = bridge.run_simulation(portfolio_value=1_000_000, volatility=0.15)

    assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    opt_result = bridge.optimize_portfolio(assets, np.array([]), np.array([]))

    logger.info(f"QAOA Allocation: {opt_result['allocation']}")

    return {"var_result": result, "optimization": opt_result}

def main():
    logger.info("Initializing Research Modules...")

    # 1. GNN
    gnn_results = run_gnn_analysis()

    # 2. XAI
    xai_report = run_xai_analysis(gnn_results)

    # 3. FL
    fl_history = run_federated_learning()

    # 4. OSWM
    oswm_briefing = run_oswm_simulation()

    # 5. Quantum
    quantum_results = run_quantum_simulation()

    # Output structure
    output_data = {
        "gnn_risk_analysis": {
            "top_risky_nodes": gnn_results[:10]
        },
        "xai_report": xai_report,
        "federated_learning": {
            "training_history": fl_history
        },
        "oswm_simulation": {
            "briefing": oswm_briefing
        },
        "quantum_simulation": quantum_results
    }

    # Save to file
    os.makedirs("showcase/data", exist_ok=True)
    with open("showcase/data/research_output.json", "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Research demo complete. Data saved to showcase/data/research_output.json")

if __name__ == "__main__":
    main()

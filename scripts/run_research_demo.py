import sys
import os
import json
import logging
import random
import torch
import numpy as np
import asyncio

# Ensure core is in path
sys.path.append(os.getcwd())

# Imports based on verified file structure from main branch
# We use try-except to handle potential missing dependencies in a diverse environment
try:
    from core.research.gnn.engine import GraphRiskEngine
    # Explainer might be optional or in development
    try:
        from core.research.gnn.explainer import GNNExplainer
    except ImportError:
        GNNExplainer = None

    from core.research.federated_learning.fl_coordinator import FederatedCoordinator
    from core.research.oswm.inference import OSWMInference
    from core.agents.strategic_foresight_agent import StrategicForesightAgent
    from core.simulations.quantum_monte_carlo import QuantumMonteCarloBridge
    from core.xai.iqnn_cs import IQNNCS
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResearchDemo")

def run_advanced_gnn_analysis():
    logger.info("--- Starting Advanced GNN Risk Analysis (GAT + Explainer) ---")
    try:
        # Use GAT model for better topological feature extraction
        engine = GraphRiskEngine(model_type="GAT")
        logger.info(f"Graph Engine initialized with {engine.num_nodes} nodes.")
        logger.info("Model Architecture: Graph Attention Network (GAT)")

        risk_scores = engine.predict_risk()

        # Explain the riskiest node 
        if risk_scores:
            sorted_risk = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
            top_risky_node, top_score = sorted_risk[0]

            logger.info(f"Explaining top risky node: {top_risky_node} (Score: {top_score:.4f})")
            
            # Use the merged Explain method if available
            if hasattr(engine, 'explain_risk'):
                mask_adj, mask_feat = engine.explain_risk(top_risky_node)

                if mask_feat is not None:
                    # Summarize feature importance from the mask tensor
                    # Taking the mean importance across the node features
                    avg_importance = torch.mean(mask_feat, dim=0).detach().numpy()
                    logger.info(f"  Explanation Generated. Top Feature Indices: {np.argsort(avg_importance)[-3:]}")
            
            return sorted_risk
        return []

    except Exception as e:
        logger.error(f"GNN Analysis failed: {e}", exc_info=True)
        return []

def run_xai_analysis(gnn_results):
    logger.info("--- Starting XAI Analysis (IQNN-CS) ---")
    try:
        # Simulate Explainable AI for the GNN results
        feature_names = ["Type", "RiskScore", "Debt", "Impact", "Sentiment"]
        iqnn = IQNNCS(input_dim=5, num_classes=2, feature_names=feature_names)

        # Generate synthetic attributions based on GNN risk scores
        # This mocks the data that would normally come from the model
        if not gnn_results:
             gnn_results = [("MockBank_A", 0.8), ("MockHedge_B", 0.3)]

        for node, score in gnn_results:
            if score > 0.7:
                attr = np.array([0.1, 0.1, 0.5, 0.3, 0.0]) + np.random.normal(0, 0.05, 5)
                predicted_class = 1
            else:
                attr = np.array([0.1, 0.1, 0.1, 0.1, 0.6]) + np.random.normal(0, 0.05, 5)
                predicted_class = 0

            iqnn.record_prediction(np.random.rand(5), predicted_class, attr)

        report = iqnn.generate_explanation_report()
        logger.info("XAI Report Generated.")
        return report
    except Exception as e:
        logger.error(f"XAI Analysis failed: {e}")
        return {}

def run_federated_learning_simulation():
    logger.info("--- Starting FinGraphFL Simulation (Privacy + MSGuard) ---")
    try:
        # Simulate 3 banks with different sector biases
        client_configs = [
            {"id": "Bank_Tech (High Vol)", "sector_bias": {1: 1.0}},
            {"id": "Bank_Energy (High Debt)", "sector_bias": {0: 1.0}},
            {"id": "Bank_Retail (Stable)", "sector_bias": {0: -0.5, 1: -0.5}}
        ]

        # Initialize Coordinator in FinGraphFL mode
        coordinator = FederatedCoordinator(
            num_clients=3, 
            input_dim=10, 
            client_configs=client_configs,
            mode="FinGraphFL" 
        )

        history = []
        for round_num in range(1, 4): # Short run for demo
            loss, acc = coordinator.run_round(round_num)
            history.append({"round": round_num, "loss": loss, "accuracy": acc})

        logger.info("FinGraphFL Simulation Complete.")
        return history
    except Exception as e:
        logger.error(f"Federated Learning failed: {e}")
        return []

def run_oswm_simulation():
    logger.info("--- Starting One-Shot World Model Simulation ---")
    try:
        # Initialize the Agent (which handles OSWM pretraining internally)
        agent = StrategicForesightAgent(config={})
        
        # Generate a Strategic Market Briefing
        # execute is likely async, so we run it in the event loop
        briefing = asyncio.run(agent.execute(ticker="SPY", horizon=10))

        logger.info(f"Strategic Briefing Status: {briefing.get('status', 'UNKNOWN')}")
        if 'narrative' in briefing:
            logger.info(f"Narrative: {briefing['narrative']}")

        return briefing
    except Exception as e:
        logger.error(f"OSWM Simulation failed: {e}")
        # Return a mock if real execution fails
        return {"status": "FAILED", "error": str(e), "trajectory": []}

def run_quantum_simulation():
    logger.info("--- Starting Quantum Monte Carlo Simulation ---")
    try:
        bridge = QuantumMonteCarloBridge()

        result = bridge.run_simulation(portfolio_value=1_000_000, volatility=0.15)

        assets = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        opt_result = bridge.optimize_portfolio(assets, np.array([]), np.array([]))

        logger.info(f"QAOA Allocation: {opt_result.get('allocation', {})}")

        return {"var_result": result, "optimization": opt_result}
    except Exception as e:
        logger.error(f"Quantum Simulation failed: {e}")
        return {}

def main():
    logger.info("Initializing Advanced Research Modules...")

    # 1. Advanced GNN (GAT + Explainer)
    gnn_results = run_advanced_gnn_analysis()

    # 2. XAI (IQNN-CS)
    xai_report = run_xai_analysis(gnn_results)

    # 3. FinGraphFL (Federated Learning with Privacy)
    fl_history = run_federated_learning_simulation()

    # 4. OSWM (One-Shot World Model)
    oswm_briefing = run_oswm_simulation()

    # 5. Quantum (QAOA)
    quantum_results = run_quantum_simulation()

    # Output structure
    output_data = {
        "gnn_risk_analysis": {
            "model": "GAT",
            "top_risky_nodes": gnn_results[:10] if gnn_results else []
        },
        "xai_report": xai_report,
        "federated_learning": {
            "mode": "FinGraphFL",
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
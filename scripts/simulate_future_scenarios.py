import json
import logging
import os
import sys
from datetime import datetime

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.v23_graph_engine.simulation_engine import CrisisSimulationEngine
from core.engine.unified_knowledge_graph import UnifiedKnowledgeGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FutureSim")

def main():
    logger.info("Initializing World Model for Simulation...")
    ukg = UnifiedKnowledgeGraph()
    engine = CrisisSimulationEngine(ukg.graph)

    # Define Scenarios
    scenarios = [
        # Historic
        {
            "id": "HIST-001",
            "title": "2008 Global Financial Crisis",
            "type": "Historic",
            "shocks": [
                {"target": "FinancialSector", "change": "-40%", "category": "Sector"},
                {"target": "RealEstate", "change": "-30%", "category": "AssetClass"},
                {"target": "CreditRating", "change": "-50%", "category": "Metric"}
            ]
        },
        {
            "id": "HIST-002",
            "title": "2020 COVID-19 Pandemic",
            "type": "Historic",
            "shocks": [
                {"target": "ConsumerDiscretionary", "change": "-35%", "category": "Sector"},
                {"target": "TechnologySector", "change": "+25%", "category": "Sector"},
                {"target": "GDP", "change": "-5%", "category": "Macro"}
            ]
        },
        # Future
        {
            "id": "FUT-001",
            "title": "2027 AI Singularity Shock",
            "type": "Future",
            "shocks": [
                {"target": "TechnologySector", "change": "+150%", "category": "Sector"},
                {"target": "LaborMarket", "change": "-20%", "category": "Macro"},
                {"target": "EnergySector", "change": "+40%", "category": "Sector"}
            ]
        },
        {
            "id": "FUT-002",
            "title": "2030 Climate Adaptation Event",
            "type": "Future",
            "shocks": [
                {"target": "InsuranceSector", "change": "-60%", "category": "Sector"},
                {"target": "RealEstate", "change": "-25%", "category": "AssetClass"},
                {"target": "GreenTech", "change": "+80%", "category": "Sector"}
            ]
        }
    ]

    results = []

    for sc in scenarios:
        logger.info(f"Running Scenario: {sc['title']}")

        # Inject scenario into engine's active list manually since we aren't loading MD files
        engine.active_scenarios.append(sc)

        try:
            sim_result = engine.run_simulation(sc['id'])
            # Enrich result
            sim_result['type'] = sc['type']
            sim_result['timestamp'] = datetime.now().isoformat()
            results.append(sim_result)
        except Exception as e:
            logger.error(f"Failed to run {sc['title']}: {e}")

    output_path = "showcase/data/future_scenarios.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Simulation complete. Saved to {output_path}")

if __name__ == "__main__":
    main()

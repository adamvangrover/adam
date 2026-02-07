import os
import sys
import json
import logging

# Ensure repo root is in path
sys.path.append(os.getcwd())

from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine

OUTPUT_FILE = "showcase/data/mock_scenario_lab_data.json"

def main():
    logging.basicConfig(level=logging.INFO)
    print("Initializing Generative Scenario Lab...")

    engine = GenerativeRiskEngine()

    # Generate batch
    # We generate a mix of regimes for the showcase
    scenarios = []

    # 50 Normal
    scenarios.extend(engine.generate_scenarios(n_samples=50, regime="normal"))

    # 30 Stress
    scenarios.extend(engine.generate_scenarios(n_samples=30, regime="stress"))

    # 20 Crash
    scenarios.extend(engine.generate_scenarios(n_samples=20, regime="crash"))

    output = {
        "generated_at": str(os.times()),
        "count": len(scenarios),
        "scenarios": [s.model_dump() for s in scenarios]
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Scenario Lab data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

import json
import os
import sys
from datetime import datetime

# Add repo root to path
sys.path.append(os.getcwd())

from core.risk.generative_risk_engine import GenerativeRiskEngine

def generate_scenario_data():
    engine = GenerativeRiskEngine()

    # Generate batch of 500 scenarios
    scenarios = engine.generate_batch(n=500)

    # Convert to dict
    data = {
        "metadata": {
            "total_scenarios": len(scenarios),
            "generated_at": datetime.now().isoformat(),
            "engine_version": "v2.4"
        },
        "scenarios": [s.model_dump() for s in scenarios]
    }

    filepath = "showcase/data/mock_scenario_lab_data.json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Generated scenario data at {filepath}")

if __name__ == "__main__":
    generate_scenario_data()

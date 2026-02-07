import os
import sys
import json
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
except ImportError as e:
    print(f"Error importing engine: {e}")
    sys.exit(1)

OUTPUT_FILE = "showcase/data/mock_scenario_lab_data.json"

def serialize_scenario(scenario):
    """Helper to handle Pydantic V1/V2 serialization differences."""
    if hasattr(scenario, 'model_dump'):
        return scenario.model_dump()
    elif hasattr(scenario, 'dict'):
        return scenario.dict()
    else:
        return scenario.__dict__

def run_generative_lab():
    logging.basicConfig(level=logging.INFO)
    print("Initializing Generative Scenario Lab...")

    try:
        engine = GenerativeRiskEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    all_scenarios = []
    
    # Define regimes and sample counts
    # We use a weighted distribution to provide a realistic mix for the showcase UI
    regime_counts = {
        "normal": 50,
        "stress": 30,
        "crash": 20
    }

    for regime, count in regime_counts.items():
        print(f"Generating {count} scenarios for regime: {regime}...")
        try:
            scenarios = engine.generate_scenarios(n_samples=count, regime=regime)
            for s in scenarios:
                s_dict = serialize_scenario(s)
                all_scenarios.append(s_dict)
        except Exception as e:
            print(f"Error generating scenarios for {regime}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    final_output = {
        "metadata": {
            "engine": "APEX Generative Risk Engine v23.5",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_scenarios": len(all_scenarios),
            "regime_distribution": regime_counts
        },
        "scenarios": all_scenarios
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Scenario Lab data generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    run_generative_lab()
import sys
import os
import json
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.vertical_risk_agent.generative_risk import GenerativeRiskEngine
except ImportError as e:
    print(f"Error importing engine: {e}")
    sys.exit(1)

def serialize_scenario(scenario):
    # Pydantic V1/V2 compatibility
    if hasattr(scenario, 'model_dump'):
        return scenario.model_dump()
    elif hasattr(scenario, 'dict'):
        return scenario.dict()
    else:
        return scenario.__dict__

def run_generative_lab():
    print("Initializing Generative Risk Engine...")
    try:
        engine = GenerativeRiskEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        # Fallback if engine fails (e.g. missing torch)
        return

    all_scenarios = []
    regimes = ["normal", "stress", "crash"]

    for regime in regimes:
        print(f"Generating scenarios for regime: {regime}...")
        try:
            scenarios = engine.generate_scenarios(n_samples=50, regime=regime)
            for s in scenarios:
                s_dict = serialize_scenario(s)
                # Ensure float values are JSON serializable (numpy floats vs python floats)
                # Pydantic usually handles this, but let's be safe if risk_factors are numpy
                all_scenarios.append(s_dict)
        except Exception as e:
            print(f"Error generating scenarios for {regime}: {e}")

    output_dir = "showcase/data"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "mock_scenario_lab_data.json")

    final_output = {
        "metadata": {
            "engine": "APEX Generative Risk Engine v23.5",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_scenarios": len(all_scenarios)
        },
        "scenarios": all_scenarios
    }

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=4)

    print(f"Generative Lab data generated at {output_file}")

if __name__ == "__main__":
    run_generative_lab()

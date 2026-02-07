import os
import sys
import json
import logging

# Ensure repo root is in path
sys.path.append(os.getcwd())

from core.simulations.financial_wargame_engine import FinancialWargameEngine

OUTPUT_FILE = "showcase/data/wargame_log.json"

def main():
    print("Initializing Cyber-Financial Wargame...")
    engine = FinancialWargameEngine()

    print("Running Simulation...")
    state = engine.run_simulation(max_turns=20)

    print(f"Simulation Complete. Winner: {state.winner}")
    print(f"Final Health: {state.financial_health:.1f}%")

    # Serialize
    output = state.model_dump()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Game log saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

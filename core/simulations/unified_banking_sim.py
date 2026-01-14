import json
import logging
import os
import sys
from typing import List, Dict, Any

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.simulations.world_model import WorldModelEngine, MarketState

# Configuration
TWIN_FILE = 'data/virtual_twins/jpm_unified_banking.json'
OUTPUT_FILE = 'showcase/data/unified_banking_scenarios.json'
SIMULATION_STEPS = 100

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedBankingSim")

def load_twin(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

def run_scenario(scenario_name: str, twin_data: Dict[str, Any], initial_temp: float, shock: Dict[str, Any] = None) -> Dict[str, Any]:
    logger.info(f"--- Running Scenario: {scenario_name} ---")

    entities = twin_data.get('entities', [])
    asset_map = {e['id']: e for e in entities if 'initial_value' in e}
    assets = list(asset_map.keys())

    engine = WorldModelEngine(assets)

    initial_prices = {aid: asset_map[aid]['initial_value'] for aid in assets}
    initial_flows = {aid: 0.0 for aid in assets}

    init_state = MarketState(
        timestamp=0,
        asset_prices=initial_prices,
        market_temperature=initial_temp,
        capital_flows=initial_flows
    )

    trajectory = []
    if shock:
        # Run counterfactual if shock provided
        # Note: WorldModelEngine.run_counterfactual returns a full trajectory
        trajectory = engine.run_counterfactual(init_state, shock)
    else:
        # Baseline run
        trajectory = engine.generate_trajectory(init_state, steps=SIMULATION_STEPS)

    # Format history for frontend: Array of { timestamp: T, values: {id: val}, flows: {id: val}, temp: X }
    history = []
    for state in trajectory:
        history.append({
            "timestamp": state.timestamp,
            "values": {k: round(v, 2) for k, v in state.asset_prices.items()},
            "flows": {k: round(v, 2) for k, v in state.capital_flows.items()},
            "temp": round(state.market_temperature, 2)
        })

    return history

def run_simulation():
    logger.info(f"Loading twin definition from {TWIN_FILE}...")
    try:
        twin_data = load_twin(TWIN_FILE)
    except FileNotFoundError:
        logger.error(f"Twin definition file not found: {TWIN_FILE}")
        return

    scenarios_output = {}

    # 1. Baseline Growth
    scenarios_output["Baseline"] = run_scenario(
        "Baseline Growth",
        twin_data,
        initial_temp=0.5,
        shock=None
    )

    # 2. Liquidity Crunch (High Temp, Shock to CIB)
    scenarios_output["Liquidity Crunch"] = run_scenario(
        "Liquidity Crunch",
        twin_data,
        initial_temp=2.0,
        shock={"shock_asset": "3", "shock_magnitude": -50.0} # Shock to CIB
    )

    # 3. Cyber Event (Moderate Temp, Shock to Holding Co & Infra)
    scenarios_output["Cyber Event"] = run_scenario(
        "Cyber Event",
        twin_data,
        initial_temp=1.2,
        shock={"shock_asset": "100", "shock_magnitude": -40.0} # Shock to Onyx
    )

    final_output = {
        "metadata": {
            "twin_name": twin_data.get('name'),
            "description": twin_data.get('description')
        },
        "entities": twin_data.get('entities', []),
        "relationships": twin_data.get('relationships', []),
        "scenarios": scenarios_output
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"All scenarios complete. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_simulation()

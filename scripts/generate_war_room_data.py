import sys
import os
import json
import asyncio
from datetime import datetime

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulations.sovereign_conflict import SovereignConflictSimulation
from core.agents.strategic_foresight_agent import StrategicForesightAgent

OUTPUT_PATH = "showcase/data/war_room_data.json"

async def generate_data():
    print("Initializing Sovereign Conflict Simulation...")
    sim_engine = SovereignConflictSimulation()

    print("Initializing Strategic Foresight Agent...")
    # Config can be minimal for this offline generation
    agent = StrategicForesightAgent(config={"agent_id": "strategic_foresight_01"})

    output_data = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.2",
            "description": "Pre-computed simulation data for Strategic War Room showcase. Includes Monte Carlo & Critique Swarm."
        },
        "scenarios": {}
    }

    scenarios_to_run = [
        {"name": "Semiconductor Blockade", "intensity": 8, "id": "sim_semi_blockade"},
        {"name": "Energy Shock", "intensity": 6, "id": "sim_energy_shock"},
        {"name": "Cyber Infrastructure Attack", "intensity": 9, "id": "sim_cyber_attack"},
        {"name": "Quantum Decryption Event", "intensity": 10, "id": "sim_quantum_event"}
    ]

    for req in scenarios_to_run:
        print(f"Running scenario: {req['name']}...")

        # 1. Run Monte Carlo Simulation
        # Using 20 iterations for speed in this demo script
        sim_result = sim_engine.run_monte_carlo(req['name'], intensity=req['intensity'], iterations=20)

        # 2. Agent Analysis
        briefing = await agent.execute(simulation_data=sim_result)

        # 3. Store Result
        # We store the "representative" timeline as the main one for the playback
        # But we pass the stats to the frontend too
        output_data["scenarios"][req['id']] = {
            "simulation": {
                "scenario": sim_result["scenario"],
                "timeline": sim_result["representative_timeline"],
                "sector_impact": sim_result["representative_sector_impact"],
                "statistics": sim_result["statistics"],
                "sector_stats": sim_result["sector_stats"]
            },
            "briefing": briefing
        }
        print(f"Completed {req['name']}.")

    # Save to file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Data written to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(generate_data())

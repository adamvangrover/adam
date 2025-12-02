import asyncio
import logging
import sys
import os

# Ensure core is in path
sys.path.append(os.getcwd())

from core.engine.crisis_simulation_graph import crisis_simulation_app
from core.engine.states import init_crisis_state
from core.agents.reflector_agent import ReflectorAgent
from core.agents.red_team_agent import RedTeamAgent

async def main():
    print("Verifying v23 Agent Expansion...")

    # 1. Verify Crisis Graph
    print("1. Testing CrisisSimulationGraph instantiation...")
    assert crisis_simulation_app is not None
    state = init_crisis_state("Interest rates up 5%", {})
    print("   State initialized:", state["human_readable_status"])

    # 2. Verify Reflector Agent
    print("2. Testing ReflectorAgent instantiation...")
    reflector = ReflectorAgent({})
    res = await reflector.execute("This is a test logic trace that is too short.")
    print("   Reflector Result:", res)
    assert "critique_notes" in res

    # 3. Verify RedTeam Agent
    print("3. Testing RedTeamAgent instantiation...")
    red_team = RedTeamAgent({})
    # We won't run it fully as it might call the graph which needs LangGraph runtime
    print("   RedTeam Agent initialized.")

    print("Verification Successful!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

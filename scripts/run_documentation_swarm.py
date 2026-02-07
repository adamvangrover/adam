import asyncio
import os
import sys
from core.agents.meta_agents.auto_architect_agent import AutoArchitectAgent
from core.agents.meta_agents.skill_harvester_agent import SkillHarvesterAgent

# Ensure repo root
sys.path.append(os.getcwd())

async def run_documentation_swarm():
    print("Initializing Documentation Swarm...")

    # 1. Auto-Architect
    architect = AutoArchitectAgent(config={"name": "AutoArch"})
    await architect.execute()
    print("Auto-Architect: Scan Complete.")

    # 2. Skill Harvester
    harvester = SkillHarvesterAgent(config={"name": "Harvester"})
    await harvester.execute()
    print("Skill Harvester: Registry Updated.")

if __name__ == "__main__":
    asyncio.run(run_documentation_swarm())

import sys
import os
import time
import random
import logging
import asyncio
from datetime import datetime

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.swarms.memory_matrix import MemoryMatrix
from core.engine.consensus_engine_v2 import ConsensusEngineV2
from core.agents.repo_knowledge_agent import RepoKnowledgeAgent

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SwarmSimulation")

def simulate_agent_activity(memory: MemoryMatrix):
    """
    Simulates various agents publishing insights to the Swarm Memory.
    """
    agents = [
        ("MarketSentimentAgent", ["Bullish", "Bearish", "Neutral"]),
        ("MacroEconomist", ["Inflation", "Rates", "Growth"]),
        ("RiskOfficer", ["Volatility", "Liquidity", "Geopolitics"]),
        ("TechAnalyst", ["AI", "Semiconductors", "SaaS"]),
        ("EnergyAnalyst", ["Oil", "Green Energy", "Nuclear"])
    ]

    # Pick a random agent
    agent_name, topics = random.choice(agents)
    topic = random.choice(topics)

    # Generate random insight content based on topic
    sentiment_drivers = [
        "showing strong momentum",
        "facing headwinds",
        "stabilizing after recent moves",
        "breaking out to new highs",
        "consolidating at support"
    ]

    risk_drivers = [
        "critical risk of insolvency",
        "regulatory crackdown imminent",
        "liquidity crisis unfolding",
        "bubble signals flashing red"
    ]

    # 10% chance of generating a CRITICAL RISK
    if random.random() < 0.1:
        content = f"WARNING: {topic} {random.choice(risk_drivers)}. Immediate attention required."
        confidence = random.uniform(0.85, 0.99)
    else:
        content = f"{topic} is {random.choice(sentiment_drivers)}. Outlook depends on macro factors."
        confidence = random.uniform(0.5, 0.9)

    # Write to memory
    logger.info(f"Agent {agent_name} publishing insight on {topic}...")
    memory.write_consensus(topic, content, agent_name, confidence)

async def run_loop():
    logger.info("Starting Swarm Consensus Simulation Loop...")

    memory = MemoryMatrix()
    engine = ConsensusEngineV2(memory_matrix=memory)

    # Initialize Repo Agent
    repo_agent = RepoKnowledgeAgent({"agent_id": "SystemArchitect", "trace_id": "sys-001"})
    # Manually link memory since the loop uses a shared one
    repo_agent.swarm_memory = memory

    tick_count = 0

    try:
        while True:
            tick_count += 1

            # 1. Simulate Agents feeding the memory
            # Simulate 1-3 insights per tick
            for _ in range(random.randint(1, 3)):
                simulate_agent_activity(memory)

            # 2. Run Repo Knowledge Agent (Every 5 ticks)
            if tick_count % 5 == 0 or tick_count == 1:
                logger.info("Executing RepoKnowledgeAgent scan...")
                await repo_agent.execute()

            # 3. Prune old memory (keep it fresh, e.g., last 1 hour for demo)
            memory.prune_stale_insights(max_age_hours=1)

            # 4. Run Consensus Engine
            report_path = engine.generate_report()
            logger.info(f"Consensus Cycle Complete. Report updated at {report_path}")

            # 5. Wait
            time.sleep(5) # 5 second tick

    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")

if __name__ == "__main__":
    asyncio.run(run_loop())

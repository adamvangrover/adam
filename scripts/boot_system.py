import asyncio
import logging
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.engine.swarm.neuro_worker import NeuroQuantumWorker
from core.engine.swarm.pheromone_board import PheromoneBoard

# Configure logging
logging.basicConfig(level=logging.INFO)

async def boot_full_system():
    """
    Initializes and boots core agents, triggering the version control logging protocol.
    """
    logging.info("Initiating Full System Boot Sequence...")

    # 1. Market Sentiment Agent
    sentiment_config = {
        "agent_id": "market_sentiment_01",
        "data_sources": ["news", "social", "prediction_market"],
        "sentiment_threshold": 0.6
    }
    sentiment_agent = MarketSentimentAgent(sentiment_config)
    sentiment_agent.boot()

    # 2. Fundamental Analyst Agent
    fundamental_config = {
        "agent_id": "fundamental_analyst_01",
        "persona": "Deep Value Investor"
    }
    fundamental_agent = FundamentalAnalystAgent(fundamental_config)
    fundamental_agent.boot()

    # 3. Neuro-Quantum Swarm Worker
    # Workers are distinct from AgentBase, so we manually report via their protocol if they have one,
    # or wrap them. For now, let's assume we extend the worker to use the protocol or log manually.
    # Since SwarmWorker doesn't inherit AgentBase, we'll manually use the BootProtocol mixin logic here
    # or assume we patch it. For this script, let's assume we just log it directly via the logger
    # since we didn't modify SwarmWorker base class yet.

    # Actually, let's just use the SystemBootLogger directly for the worker to show flexibility
    from core.system.system_boot_logger import SystemBootLogger, BootLogEntry
    import time

    board = PheromoneBoard()
    neuro_worker = NeuroQuantumWorker(board, role="neuro_quantum_prime")

    # Simulate worker boot logic
    entry = BootLogEntry(
        timestamp=time.time(),
        agent_id=neuro_worker.id,
        status="WORKER_ONLINE",
        highest_conviction_prompt="Neuro-Quantum Lattice Initialization",
        conviction_score=0.99
    )
    SystemBootLogger.log_boot(entry)

    logging.info("System Boot Sequence Complete.")

if __name__ == "__main__":
    asyncio.run(boot_full_system())

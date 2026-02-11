import asyncio
import random
import logging
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager

# Import from the mesh bridge
try:
    from core.v30_architecture.python_intelligence.bridge.neural_mesh import app, NeuralPacket, emit_packet
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from core.v30_architecture.python_intelligence.bridge.neural_mesh import app, NeuralPacket, emit_packet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SwarmRunner")

# --- Agent Definitions ---

try:
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent
    from core.v30_architecture.python_intelligence.agents.quantitative_analyst import QuantitativeAnalyst
    from core.v30_architecture.python_intelligence.agents.risk_guardian import RiskGuardian
    from core.v30_architecture.python_intelligence.agents.market_scanner import MarketScanner
except ImportError:
    # Fallback for local execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from base_agent import BaseAgent
    from quantitative_analyst import QuantitativeAnalyst
    from risk_guardian import RiskGuardian
    from market_scanner import MarketScanner

class SystemHealth(BaseAgent):
    def __init__(self):
        super().__init__("SystemSupervisor", "operations")

    async def run(self):
        while True:
            try:
                payload = {
                    "cpu_usage": round(random.uniform(20, 80), 1),
                    "memory_usage": round(random.uniform(30, 60), 1),
                    "active_agents": 4,
                    "mesh_latency_ms": int(random.uniform(5, 50))
                }

                await self.emit("system_status", payload)

            except Exception as e:
                logger.error(f"Error in SystemHealth: {e}")

            await asyncio.sleep(5.0)

# --- Swarm Orchestration ---

agents = [MarketScanner(), RiskGuardian(), SystemHealth(), QuantitativeAnalyst()]

@app.on_event("startup")
async def start_swarm():
    logger.info("Initializing V30 Swarm Agents...")
    for agent in agents:
        asyncio.create_task(agent.run())
        logger.info(f"Launched {agent.name}")

if __name__ == "__main__":
    logger.info("Starting Swarm Runner & Neural Mesh v2...")
    uvicorn.run(app, host="0.0.0.0", port=8001)

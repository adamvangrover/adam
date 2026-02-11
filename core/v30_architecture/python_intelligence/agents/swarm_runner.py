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
except ImportError:
    # Fallback for local execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from base_agent import BaseAgent
    from quantitative_analyst import QuantitativeAnalyst
    from risk_guardian import RiskGuardian

class MarketScanner(BaseAgent):
    def __init__(self):
        super().__init__("MarketScanner-V1", "data_acquisition")

    async def run(self):
        assets = ["BTC-USD", "SPY", "VIX", "ETH-USD"]
        prices = {k: random.uniform(100, 50000) for k in assets}

        while True:
            try:
                asset = random.choice(assets)
                change = random.uniform(-2.5, 2.5)
                prices[asset] *= (1 + change/100)

                payload = {
                    "symbol": asset,
                    "price": round(prices[asset], 2),
                    "change_pct": round(change, 2),
                    "volume": int(random.uniform(1000, 100000))
                }

                await self.emit("market_data", payload)

            except Exception as e:
                logger.error(f"Error in MarketScanner: {e}")

            await asyncio.sleep(random.uniform(0.5, 1.5))


class SystemHealth(BaseAgent):
    def __init__(self):
        super().__init__("SystemSupervisor", "operations")

    async def run(self):
        while True:
            try:
                payload = {
                    "cpu_usage": round(random.uniform(20, 80), 1),
                    "memory_usage": round(random.uniform(30, 60), 1),
                    "active_agents": 3,
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

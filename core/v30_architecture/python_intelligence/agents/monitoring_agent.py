import asyncio
import random
import logging
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager

# Import from the bridge (assuming running from root or as module)
# Adjust import based on execution context
try:
    from core.v30_architecture.python_intelligence.bridge.neural_link import app, Thought, emit_thought
except ImportError:
    # Fallback for direct execution if path not set
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from core.v30_architecture.python_intelligence.bridge.neural_link import app, Thought, emit_thought

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonitoringAgent")

class MonitoringAgent:
    def __init__(self, name="Sentinel-V30"):
        self.name = name

    async def run_loop(self):
        logger.info(f"{self.name} initialized. Starting heartbeat...")
        while True:
            try:
                # Synthetic data generation
                actions = [
                    "Scanning market data feeds...",
                    "Optimizing portfolio allocation...",
                    "Analyzing sentiment vectors...",
                    "Detecting market anomalies...",
                    "Rebalancing asset weights...",
                    "Synchronizing with Quantum Core...",
                    "Validating risk metrics..."
                ]
                content = random.choice(actions)
                conviction = round(random.uniform(0.1, 0.99), 2)

                thought = Thought(
                    timestamp=datetime.utcnow().isoformat(),
                    agent_name=self.name,
                    content=content,
                    conviction_score=conviction
                )

                # Send to Neural Link
                await emit_thought(thought)
                logger.info(f"Emitted: {content} ({conviction})")

            except Exception as e:
                logger.error(f"Error in agent loop: {e}")

            await asyncio.sleep(2)

# Create agent instance
agent = MonitoringAgent()

# Register agent loop as a startup event for the FastAPI app
@app.on_event("startup")
async def start_agent():
    asyncio.create_task(agent.run_loop())

if __name__ == "__main__":
    logger.info("Starting V30 Neural Link + Monitoring Agent...")
    # Run the FastAPI app which now includes the agent loop
    uvicorn.run(app, host="0.0.0.0", port=8000)

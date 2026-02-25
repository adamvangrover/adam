import asyncio
import random
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonitoringAgent")

# Ensure we can import from the bridge by adding the repo root to sys.path
# This assumes the script is run from the repo root or its own directory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from core.v30_architecture.python_intelligence.bridge.neural_link import app, emitter, Thought
except ImportError:
    # Fallback for relative imports if run as a module
    sys.path.append(os.path.join(repo_root, "core", "v30_architecture", "python_intelligence", "bridge"))
    from neural_link import app, emitter, Thought

class MonitoringAgent:
    def __init__(self, name="Watchtower-V30"):
        self.name = name
        self.running = False

    async def start(self):
        self.running = True
        logger.info(f"{self.name} initialized. Starting heartbeat...")
        while self.running:
            try:
                thought = self.generate_thought()
                await emitter.broadcast(thought)
                logger.info(f"Emitted: {thought.content} ({thought.conviction_score})")
            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
            
            await asyncio.sleep(2)

    def stop(self):
        self.running = False

    def generate_thought(self) -> Thought:
        actions = [
            "Scanning market data feeds...",
            "Optimizing portfolio allocation...",
            "Analyzing sentiment vectors...",
            "Detecting market anomalies...",
            "Rebalancing asset weights...",
            "Synchronizing with Quantum Core...",
            "Validating risk metrics...",
            "Verifying data integrity..."
        ]

        content = random.choice(actions)
        # Conviction score between 0 and 100
        conviction = round(random.uniform(50, 99), 1)

        # Determine color/status implicitly by score (low/high)
        # For demo, let's make some 'critical'
        if random.random() < 0.2:
            conviction = round(random.uniform(10, 40), 1)
            content = f"WARNING: {content} Low confidence."

        return Thought(
            timestamp=datetime.now().isoformat(),
            agent_name=self.name,
            content=content,
            conviction_score=conviction
        )

agent = MonitoringAgent()

@app.on_event("startup")
async def startup_event():
    # Start the agent loop as a background task
    asyncio.create_task(agent.start())

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {agent.name} and Neural Link...")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
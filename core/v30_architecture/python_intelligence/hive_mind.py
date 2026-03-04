import asyncio
import random
import sys
import os
import json
from datetime import datetime
from typing import List, Dict

# Ensure we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Try imports with fallbacks
try:
    from core.v30_architecture.python_intelligence.bridge.neural_link import app, emitter, Thought
    from core.v30_architecture.python_intelligence.agents.code_weaver import CodeWeaverAgent
    from core.v30_architecture.python_intelligence.agents.news_bot import NewsBotAgent
except ImportError:
    # Fallback for relative paths if module structure is tricky
    sys.path.append(os.path.join(repo_root, "core/v30_architecture/python_intelligence"))
    from bridge.neural_link import app, emitter, Thought
    from agents.code_weaver import CodeWeaverAgent
    from agents.news_bot import NewsBotAgent

# --- WRAPPERS FOR EXISTING AGENTS ---

class ConnectedNewsBot(NewsBotAgent):
    """Wraps the legacy NewsBot to emit thoughts to the Neural Link."""
    def __init__(self):
        super().__init__()
        self.name = "NewsBot-V30"

    async def run_loop(self):
        while True:
            # Simulate processing time
            await asyncio.sleep(random.uniform(3, 8))

            # Use the base class logic (simulated here since run_cycle is sync)
            # In a real async migration, we'd refactor run_cycle to be async.
            # Here we wrap the sync call.
            try:
                # Mocking the output for visual variety since the original base class has limited mock data
                topics = ["AAPL", "TSLA", "NVDA", "BTC", "ETH", "Macro", "Fed"]
                topic = random.choice(topics)
                sentiment = random.uniform(-0.9, 0.9)

                content = f"Analyzing sentiment for {topic}..."
                if abs(sentiment) > 0.6:
                    trend = "BULLISH" if sentiment > 0 else "BEARISH"
                    content = f"ALERT: Strong {trend} signal on {topic} ({sentiment:.2f})"

                thought = Thought(
                    timestamp=datetime.now().isoformat(),
                    agent_name=self.name,
                    content=content,
                    conviction_score=round(abs(sentiment) * 100, 1)
                )
                await emitter.broadcast(thought)
            except Exception as e:
                print(f"NewsBot Error: {e}")

class ConnectedCodeWeaver(CodeWeaverAgent):
    """Wraps the CodeWeaver to emit maintenance thoughts."""
    def __init__(self):
        super().__init__()
        self.name = "CodeWeaver-V30"

    async def run_loop(self):
        while True:
            await asyncio.sleep(random.uniform(5, 12))

            actions = [
                "Scanning for technical debt...",
                "Optimizing import structure...",
                "Refactoring legacy v19 modules...",
                "Detecting unused variables...",
                "Verifying type hints compatibility...",
                "Generating fix PR for issue #42..."
            ]
            action = random.choice(actions)

            # Simulate finding an issue occasionally
            conviction = 95.0
            if "Scanning" in action:
                conviction = 50.0

            thought = Thought(
                timestamp=datetime.now().isoformat(),
                agent_name=self.name,
                content=action,
                conviction_score=conviction
            )
            await emitter.broadcast(thought)

# --- NEW META-AGENTS ---

class RiskManagerAgent:
    def __init__(self):
        self.name = "RiskManager-V30"

    async def run_loop(self):
        while True:
            await asyncio.sleep(random.uniform(2, 5))

            metrics = [
                ("VaR (95%)", "Safe"),
                ("Sharpe Ratio", "Optimizing"),
                ("Beta Exposure", "High"),
                ("Liquidity", "Adequate"),
                ("Drawdown", "Within Limits")
            ]
            metric, status = random.choice(metrics)

            content = f"Monitoring {metric}: {status}"
            score = random.uniform(80, 100)

            if status == "High" or status == "Optimizing":
                content += f". Adjusting parameters."
                score = random.uniform(60, 80)

            thought = Thought(
                timestamp=datetime.now().isoformat(),
                agent_name=self.name,
                content=content,
                conviction_score=round(score, 1)
            )
            await emitter.broadcast(thought)

class SwarmOrchestratorAgent:
    def __init__(self):
        self.name = "SwarmOrchestrator-V30"

    async def run_loop(self):
        while True:
            await asyncio.sleep(random.uniform(4, 10))

            tasks = [
                "Delegating alpha search to AlgoSwarm...",
                "Aggregating consensus from 12 sub-agents...",
                "Rebalancing compute resources...",
                "Syncing knowledge graph with new market data...",
                "Broadcasting strategy update to execution layer..."
            ]
            task = random.choice(tasks)

            thought = Thought(
                timestamp=datetime.now().isoformat(),
                agent_name=self.name,
                content=task,
                conviction_score=99.9  # Orchestrator is always confident
            )
            await emitter.broadcast(thought)

# --- RUNNER ---

# Instantiate agents
news_bot = ConnectedNewsBot()
code_weaver = ConnectedCodeWeaver()
risk_manager = RiskManagerAgent()
orchestrator = SwarmOrchestratorAgent()

@app.on_event("startup")
async def startup_event():
    print("Hive Mind Initializing...")
    # Launch all agent loops
    asyncio.create_task(news_bot.run_loop())
    asyncio.create_task(code_weaver.run_loop())
    asyncio.create_task(risk_manager.run_loop())
    asyncio.create_task(orchestrator.run_loop())
    print("All Agents Online.")

if __name__ == "__main__":
    import uvicorn
    print("Starting Hive Mind Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104

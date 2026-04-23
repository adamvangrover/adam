
import asyncio
import logging
from typing import Any, Dict

from core.system.v22_async.async_agent_base import AsyncAgentBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoAdaptiveAgent(AsyncAgentBase):
    """
    A demo agent to verify the Adaptive Conviction and Subscription patterns.
    """
    async def execute(self, **kwargs) -> Any:
        task = kwargs.get("task", "unknown task")
        tools = kwargs.get("relevant_tools", [])

        logger.info(f"DemoAgent Executing task: {task}")
        if tools:
            logger.info(f"Using tools: {[t['name'] for t in tools]}")

        # Simulate some work
        await asyncio.sleep(0.1)

        # Simulate state change for drift detection test
        if "induce_drift" in kwargs:
            self.context["status"] = "drifted"

        return f"Executed: {task}"

async def main():
    print("--- Starting Adaptive System Demo ---")

    # Define some dummy tools
    tools = [
        {"name": "market_search", "description": "Search for market data"},
        {"name": "email_sender", "description": "Send emails"},
        {"name": "calculator", "description": "Perform calculations"}
    ]

    config = {"tools": tools}
    agent = DemoAdaptiveAgent(config=config)

    # 1. Test Low Conviction (Hedging)
    print("\n[Test 1] Low Conviction Task")
    # "maybe" triggers the hedging heuristic
    result = await agent.adaptive_execute("maybe run a market search?")
    print(f"Result: {result}")

    # 2. Test High Conviction + Tool RAG
    print("\n[Test 2] High Conviction Task with Tool RAG")
    # Should retrieve 'market_search'
    result = await agent.adaptive_execute("run a market search for AAPL")
    print(f"Result: {result}")

    # 3. Test State Anchor & Drift
    print("\n[Test 3] State Anchor & Drift")
    agent.context["status"] = "normal"
    # Enable drift simulation
    result = await agent.adaptive_execute("long_running analysis", long_running=True, induce_drift=True)
    print(f"Result: {result}") # Should show warning

    # 4. Test Subscription Damping
    print("\n[Test 4] Subscription Damping")
    topic = "market_updates"

    def on_update(msg):
        print(f"Received update: {msg}")

    agent.subscribe_to_topic(topic, on_update)

    # Publish multiple times quickly
    agent.publish_event(topic, "UPDATE", "Price: 100")
    agent.publish_event(topic, "UPDATE", "Price: 101") # Should be dampened
    agent.publish_event(topic, "UPDATE", "Price: 102") # Should be dampened

    await asyncio.sleep(2.1) # Wait for damping to expire
    agent.publish_event(topic, "UPDATE", "Price: 105") # Should go through

    await asyncio.sleep(1)
    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(main())

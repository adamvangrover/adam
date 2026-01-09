import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock, patch

# Mock Config
sys.path.append(os.getcwd())

# Mock Environment Variables
os.environ["COHERE_API_KEY"] = "mock_key"
os.environ["OPENAI_API_KEY"] = "mock_key"

# Mock LLMPlugin to avoid API calls
sys.modules["core.llm_plugin"] = MagicMock()
from core.llm_plugin import LLMPlugin
LLMPlugin.return_value.generate.return_value = "Mock Response"

from core.engine.meta_orchestrator import MetaOrchestrator

async def verify():
    print("Initializing MetaOrchestrator...")

    # We need to mock legacy orchestrator initialization because it tries to load LLMPlugin
    # inside its __init__ which might fail if sys.modules mock isn't enough (it imports before).
    # But I set sys.modules first.

    try:
        orchestrator = MetaOrchestrator()
    except Exception as e:
        print(f"Orchestrator Init Failed: {e}")
        return

    query = "Activate scenario BEAR_CRASH"
    print(f"Sending Query: {query}")

    # This should route to MARKET_CONTROL -> MarketUpdateAgent
    result = await orchestrator.route_request(query)

    print(f"Result: {result}")

    if isinstance(result, dict) and result.get("status") == "Market Control Executed":
        print("SUCCESS: Routed to Market Control")
    else:
        print("FAILURE: Routing incorrect")

if __name__ == "__main__":
    asyncio.run(verify())

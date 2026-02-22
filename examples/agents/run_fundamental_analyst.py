import asyncio
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.schemas.agent_schema import AgentInput

async def main():
    print("--- Running Fundamental Analyst Example ---")

    # 1. Configure
    config = {
        "persona": "Senior Analyst",
        "description": "Example run",
        # Mocking peers for standalone run
        "peers": []
    }

    # 2. Instantiate
    agent = FundamentalAnalystAgent(config=config)

    # 3. Execute
    # Using 'ABC_TEST' which triggers the internal mock data in the agent's test block logic
    # In production, you'd use a real ticker like 'AAPL' and ensure DataRetrievalAgent is connected.
    input_data = AgentInput(query="ABC_TEST")

    print(f"Input: {input_data.query}")
    result = await agent.execute(input_data)

    # 4. Output
    print("\n--- Result ---")
    print(f"Confidence: {result.confidence}")
    print(f"Answer: {result.answer}")
    print(f"Metadata Keys: {list(result.metadata.keys())}")

if __name__ == "__main__":
    asyncio.run(main())

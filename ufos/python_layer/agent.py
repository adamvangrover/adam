import asyncio
import httpx
import json
import logging

# Chapter 1.2: The Intelligence Layer (Python)
# Acts as the MCP Client, consuming resources and invoking tools.

class GhostAgent:
    def __init__(self, mcp_url="http://localhost:3000"):
        self.mcp_url = mcp_url
        self.client = httpx.AsyncClient()
        self.logger = logging.getLogger("GhostAgent")
        logging.basicConfig(level=logging.INFO)

    async def connect(self):
        """Connects to the MCP Server via SSE to listen for market events."""
        self.logger.info(f"Connecting to MCP Core at {self.mcp_url}/sse")
        async with self.client.stream("GET", f"{self.mcp_url}/sse") as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    event_data = line[5:].strip()
                    await self.process_event(event_data)

    async def process_event(self, event_data):
        """Processes market data updates (Resources)."""
        # In a real scenario, this parses JSON and updates the internal context window
        self.logger.info(f"Received Market Event: {event_data}")

        # 5.2 Anomaly Detection trigger (simulated)
        # if anomaly_score > threshold:
        #     await self.handle_anomaly()

    async def execute_trade(self, symbol, quantity, side):
        """Invokes the 'execute_order' Tool on the Rust Core."""
        payload = {
            "method": "execute_order",
            "params": {
                "symbol": symbol,
                "quantity": quantity,
                "side": side
            }
        }
        response = await self.client.post(f"{self.mcp_url}/mcp", json=payload)
        self.logger.info(f"Trade Execution Response: {response.json()}")

    async def tune_strategy(self, gamma, kappa):
        """Updates the Avellaneda-Stoikov parameters."""
        # This might be a tool call or a direct PyO3 call if running locally
        pass

async def main():
    agent = GhostAgent()
    # Start listening in background
    listener = asyncio.create_task(agent.connect())

    # Simulate agent logic loop
    await asyncio.sleep(2)
    await agent.execute_trade("BTC-USD", 100, "buy")

    await listener

if __name__ == "__main__":
    asyncio.run(main())

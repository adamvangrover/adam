from core.agents.agent_base import AgentBase
from core.market_data.manager import MarketDataManager
from core.market_data.scenarios import SCENARIOS
from typing import Dict, Any, Optional
import re

class MarketUpdateAgent(AgentBase):
    """
    Agent responsible for handling dynamic market data updates.
    It interprets user commands and updates the system's Real-Time State.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        config = config or {"agent_id": "MarketUpdateAgent"}
        super().__init__(config=config, **kwargs)
        self.data_manager = MarketDataManager()

    async def execute(self, instruction: str, **kwargs):
        """
        Parses instruction and updates state.
        Supported patterns:
        - "Update [SYMBOL] to [PRICE]"
        - "Set [SYMBOL] price [PRICE]"
        - "News: [HEADLINE]"
        - "Activate scenario [NAME]"
        - "Get scenario"
        - "Simulate market"
        """
        if not instruction:
            return "No instruction provided."

        instruction = instruction.strip()

        # 1. Scenario Activation
        scenario_match = re.search(r'(?:activate|set|start)\s+scenario\s+([\w_]+)', instruction, re.IGNORECASE)
        if scenario_match:
            name = scenario_match.group(1).upper()
            if name in SCENARIOS:
                msg = self.data_manager.set_scenario(name)
                return msg
            else:
                return f"Scenario '{name}' not found. Available: {', '.join(SCENARIOS.keys())}"

        # 2. Get Scenario
        if "get scenario" in instruction.lower() or "current scenario" in instruction.lower():
            return f"Current Active Scenario: {self.data_manager.active_scenario.name}"

        # 3. Price Update Pattern
        price_match = re.search(r'(?:update|set)\s+([A-Z0-9=_.-]+)\s+(?:to|price)\s+\$?([\d,]+(?:\.\d+)?)', instruction, re.IGNORECASE)
        if price_match:
            symbol = price_match.group(1).upper()
            price = float(price_match.group(2).replace(',', ''))
            self.data_manager.update_symbol(symbol, price, source="User Agent")
            return f"Updated {symbol} to {price}"

        # 4. News Injection Pattern
        news_match = re.search(r'news:\s*(.*)', instruction, re.IGNORECASE)
        if news_match:
            headline = news_match.group(1)
            # Simple sentiment heuristic
            sentiment = "POSITIVE" if any(w in headline.lower() for w in ["soars", "up", "bull", "gain"]) else "NEGATIVE"
            self.data_manager.add_news(headline, sentiment, source="User Agent")
            return f"Injected News: {headline}"

        # 5. Simulation Trigger
        if "simulate" in instruction.lower():
            self.data_manager.simulate_step()
            return f"Triggered simulation step (Scenario: {self.data_manager.active_scenario.name})."

        return "Command not recognized. Try 'Activate scenario BULL_RALLY' or 'Update AAPL to 150'."

# Test
if __name__ == "__main__":
    import asyncio
    agent = MarketUpdateAgent()
    print(asyncio.run(agent.execute("Activate scenario BULL_RALLY")))
    print(asyncio.run(agent.execute("Simulate market")))

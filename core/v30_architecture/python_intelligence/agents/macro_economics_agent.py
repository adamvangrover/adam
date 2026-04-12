import asyncio
import random
import logging
from typing import Dict, Any

try:
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
    from core.v30_architecture.python_intelligence.agents.base_agent import BaseAgent

logger = logging.getLogger("MacroEconomicsAgent")

class MacroEconomicsAgent(BaseAgent):
    """
    A V30 agent that emits macroeconomic data and determines the overall market regime.
    """
    def __init__(self):
        super().__init__("MacroEcon-V30", "macro_analysis")

    async def run(self):
        logger.info(f"{self.name} started. Monitoring Macroeconomic factors.")
        while True:
            try:
                # Simulate fetching macro data
                await self.analyze_macro_state()
            except Exception as e:
                logger.error(f"Error in MacroEconomicsAgent loop: {e}")

            # Wait before next macro update
            await asyncio.sleep(random.uniform(45.0, 90.0))

    async def analyze_macro_state(self):
        """
        Simulates parsing macro data like CPI, Fed Funds Rate, and GDP growth.
        Emits a regime classification that other agents can use.
        """
        # Simulated data point variations
        fed_funds_rate = round(random.uniform(4.5, 5.5), 2)
        cpi_yoy = round(random.uniform(2.5, 4.0), 2)
        gdp_growth = round(random.uniform(1.0, 3.5), 2)

        # Determine regime
        regime = "NEUTRAL"
        if cpi_yoy > 3.5 and fed_funds_rate > 5.0:
            regime = "RESTRICTIVE"
        elif gdp_growth > 2.5 and cpi_yoy < 3.0:
            regime = "EXPANSIONARY"
        elif gdp_growth < 1.5:
            regime = "STAGFLATION_RISK"

        payload = {
            "fed_funds_rate": fed_funds_rate,
            "cpi_yoy": cpi_yoy,
            "gdp_growth": gdp_growth,
            "regime": regime
        }

        await self.emit("macro_update", payload)
        logger.info(f"Emitted Macro Update: Regime={regime}")

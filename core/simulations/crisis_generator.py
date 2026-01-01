import logging
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from core.llm_plugin import LLMPlugin

logger = logging.getLogger(__name__)

class CrisisScenario(BaseModel):
    title: str
    trigger_event: str
    economic_impact: str
    sector_impacts: List[Dict[str, str]]
    probability: str

class CrisisGenerator:
    """
    Generative engine for creating 'Black Swan' crisis scenarios.
    Uses LLM hallucination constructively to imagine stress tests.
    """

    def __init__(self):
        self.llm = LLMPlugin()

    async def generate_scenario(self, macro_params: Dict[str, Any]) -> CrisisScenario:
        """
        Generates a detailed crisis scenario based on macro-economic inputs.
        """
        prompt = f"""
        Generate a detailed "Black Swan" financial crisis scenario based on the following parameters:
        {macro_params}

        The scenario should be plausible but extreme.
        Include:
        1. A catchy Title.
        2. The Trigger Event (Geopolitical, Tech failure, Natural disaster).
        3. Global Economic Impact.
        4. Specific impacts on Tech, Finance, and Energy sectors.
        5. Estimated Probability (e.g., "1 in 50 years").

        Output strictly as JSON matching the CrisisScenario schema.
        """

        try:
            loop = asyncio.get_running_loop()
            result, _ = await loop.run_in_executor(
                None,
                lambda: self.llm.generate_structured(prompt, CrisisScenario)
            )
            return result
        except Exception as e:
            logger.error(f"Crisis Generation failed: {e}")
            raise

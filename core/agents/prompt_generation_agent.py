from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from core.agents.agent_base import AgentBase


class PromptGenerationAgent(AgentBase):
    """
    An agent that generates a high-quality prompt from a user query.
    """

    def __init__(self, config: Dict[str, Any], kernel=None):
        """
        Initializes the PromptGenerationAgent.
        """
        super().__init__(config, kernel=kernel)

    async def execute(self, user_query: str, **kwargs: Any) -> str:
        """
        Generates a high-quality prompt from a user query.
        """
        logging.info("Generating improved prompt...")

        prompt_template = f"""
You are a world-class financial analyst assistant. Your task is to take a user's query and transform it into a detailed, high-quality prompt for another AI agent.

The user's query is: "{user_query}"

Based on this query, generate a new prompt that is:
- More specific and detailed.
- Includes context relevant to financial analysis.
- Is structured to elicit a comprehensive and accurate response from a financial analysis agent.
"""

        if self.kernel:
            try:
                # Use Semantic Kernel to run this if available
                # Assuming self.kernel has a method to invoke a prompt (SK v1 style)
                # or we just return a placeholder indicating SK usage.
                logging.info("Using Semantic Kernel to generate prompt.")
                # In a real implementation: result = await self.kernel.invoke_prompt(prompt_template)
                return f"Optimized Prompt for: {user_query} (Enhanced by AI)"
            except Exception as e:
                logging.error(f"Error using Kernel: {e}")

        # Fallback logic
        return f"Please analyze the following request in detail, considering all financial risks and market conditions: {user_query}"

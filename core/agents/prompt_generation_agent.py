# core/agents/prompt_generation_agent.py

from core.agents.agent_base import AgentBase
from core.llm.base_llm_engine import BaseLLMEngine
from typing import Any, Dict

class PromptGenerationAgent(AgentBase):
    """
    An agent that generates a high-quality prompt from a user query.
    """

    def __init__(self, config: Dict[str, Any], llm_engine: BaseLLMEngine):
        """
        Initializes the PromptGenerationAgent.
        """
        super().__init__(config)
        self.llm_engine = llm_engine

    async def execute(self, user_query: str, **kwargs: Any) -> str:
        """
        Generates a high-quality prompt from a user query.
        """
        # Create a prompt for the LLM to generate a better prompt
        prompt_for_llm = f\"\"\"
You are a world-class financial analyst assistant. Your task is to take a user's query and transform it into a detailed, high-quality prompt for another AI agent.

The user's query is: "{user_query}"

Based on this query, generate a new prompt that is:
- More specific and detailed.
- Includes context relevant to financial analysis.
- Is structured to elicit a comprehensive and accurate response from a financial analysis agent.

The generated prompt should be a single block of text.
\"\"\"

        # Use the LLM to generate the new prompt
        tuned_prompt = await self.llm_engine.generate_response(prompt=prompt_for_llm)

        return tuned_prompt

# core/agents/knowledge_contribution_agent.py

from core.agents.agent_base import AgentBase
from core.llm.base_llm_engine import BaseLLMEngine
from typing import Any, Dict
import json

class KnowledgeContributionAgent(AgentBase):
    """
    An agent that extracts key findings from a report and formats them as structured data.
    """

    def __init__(self, config: Dict[str, Any], llm_engine: BaseLLMEngine):
        """
        Initializes the KnowledgeContributionAgent.
        """
        super().__init__(config)
        self.llm_engine = llm_engine

    async def execute(self, final_report: str, **kwargs: Any) -> str:
        """
        Extracts key findings from a report and formats them as structured data.
        """
        # Create a prompt for the LLM to extract key findings
        prompt_for_llm = f"""
You are a data extraction and structuring expert. Your task is to analyze a financial report and extract the key findings.

The financial report is:
---
{final_report}
---

Please extract the key findings from this report and format them as a JSON object. The JSON object should have the following structure:
{{
  "key_findings": [
    {{
      "finding": "A brief summary of the finding.",
      "supporting_evidence": [
        "A list of supporting evidence from the report."
      ],
      "confidence_score": "A score from 0 to 1 indicating the confidence in the finding."
    }}
  ]
}}

The output should be a single JSON object.
"""

        # Use the LLM to generate the structured data
        structured_data_str = await self.llm_engine.generate_response(prompt=prompt_for_llm)

        # Parse the JSON string
        try:
            structured_data = json.loads(structured_data_str)
        except json.JSONDecodeError:
            # Handle the case where the LLM does not return valid JSON
            # For now, we'll just return an empty dictionary
            structured_data = {}

        return structured_data

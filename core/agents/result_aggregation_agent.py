# core/agents/result_aggregation_agent.py

import logging
from core.agents.agent_base import AgentBase
from core.utils.config_utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ResultAggregationAgent(AgentBase):
    """
    Combines results from multiple agents.  Initially uses simple concatenation,
    but is designed for future LLM integration.
    """

    def __init__(self, config_path="config/agents.yaml"):
        super().__init__()
        self.config = load_config(config_path)
        agent_config = self.config.get('agents', {}).get('ResultAggregationAgent', {})

        if not agent_config:
            logging.error("ResultAggregationAgent configuration not found.")
            raise ValueError("ResultAggregationAgent configuration not found.")

        self.persona = agent_config.get('persona', "Result Aggregation Agent")
        self.description = agent_config.get('description', "Combines results from multiple agents.")
        self.expertise = agent_config.get('expertise', ["data aggregation", "result summarization"])
        # self.prompt_template = agent_config.get('prompt_template', "...") # Placeholder for later


    def execute(self, results: list) -> str:
        """
        Combines results from multiple agents.

        Args:
            results: A list of results (strings, for now) from other agents.

        Returns:
            A single string representing the aggregated results.
        """
        try:
            if not results:
                return "No results to aggregate."

            aggregated_result = self._concatenate_results(results)
            logging.info(f"Results aggregated: {aggregated_result}")
            return aggregated_result

        except Exception as e:
            logging.error(f"Error in ResultAggregationAgent: {e}")
            return "An error occurred during result aggregation."

    def _concatenate_results(self, results: list) -> str:
        """
        Concatenates results with separators (basic implementation).
        """
        return "\n\n".join([str(r) for r in results]) # Handles mixed data types

    # --- Placeholder methods for future LLM integration ---
    # def _build_llm_prompt(self, results: list, context: str = "") -> str:
    #     """Builds a prompt for the LLM (future implementation)."""
    #     pass  # TODO

    # def _parse_llm_response(self, response: str) -> str:
    #     """Parses the LLM response (future implementation)."""
    #     pass  # TODO

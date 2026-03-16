# core/agents/query_understanding_agent.py

import json
import logging
from typing import Any, Dict, List, Optional

from core.agents.agent_base import AgentBase  # Assuming you have a base class for agents
from core.llm_plugin import LLMPlugin
from core.utils.token_utils import check_token_limit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QueryUnderstandingAgent(AgentBase):
    """
    An agent responsible for understanding the user's query and
    determining which other agents are relevant to answer it.
    This version incorporates LLM-based intent recognition and skill-based routing.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        """
        Initializes the QueryUnderstandingAgent.
        """
        super().__init__(config, **kwargs)  # Initialize the base class
        self.config = config  # Keep the config for access to LLM and other settings
        agent_config = self.config.get("QueryUnderstandingAgent", {})  # Load specific config
        self.persona = agent_config.get(
            "persona", "A helpful assistant specializing in task delegation."
        )
        self.expertise = agent_config.get("expertise", ["query analysis", "agent selection"])
        self.prompt_template = agent_config.get(
            "prompt_template",
            """
            You are an expert in understanding user queries and routing them to the appropriate agents.
            Consider the available agent skills and their descriptions to determine the best agent(s) to handle the query.
            Return a JSON object with the "agents" key containing a list of agent names.
            If no agents are relevant, return an empty list.

            Available Agent Skills:
            {available_agent_skills}

            User Query:
            {user_query}

            JSON Output:
            """,
        )
        self.available_agent_skills = self.get_available_agent_skills()  # MCP Skill Schemas
        self.llm_plugin = LLMPlugin()  # Access to LLM
        self.max_retries = agent_config.get("max_retries", 3)
        self.retry_delay = agent_config.get("retry_delay", 2)

    def get_available_agent_skills(self) -> str:
        """
        Gets a formatted string of available agent skills from the orchestrator's
        MCP service registry.
        """

        # Assuming the orchestrator is accessible via self.orchestrator
        # This might need to be adjusted based on how agents access the orchestrator
        if not hasattr(self, "orchestrator") or not self.orchestrator:
            # In V23 architecture, orchestration logic handles routing.
            # This method acts as a helper for self-reflection if orchestrator is injected.
            return "N/A - Orchestrator Link Inactive"

        available_skills = []
        for agent_name, skill_schema in self.orchestrator.mcp_service_registry.items():
            for skill in skill_schema["skills"]:
                available_skills.append(
                    f"- Agent: {agent_name}, Skill: {skill['name']}, Description: {skill['description']}"
                )
        return "\n".join(available_skills)

    async def execute(self, user_query: str) -> Optional[List[str]]:
        """
        Executes the query understanding process using an LLM.

        Args:
            user_query: The user's input query.

        Returns:
            A list of agent names (strings) that are deemed relevant to the query.
            Returns None if an error occurs or if the LLM fails to provide a valid output.
        """
        logging.info(f"QueryUnderstandingAgent executing on query: {user_query}")

        if not check_token_limit(user_query, self.config):
            error_message = "User query exceeds the token limit."
            logging.error(error_message)
            return None  # Indicate failure

        prompt: str = self.prompt_template.format(
            available_agent_skills=self.available_agent_skills, user_query=user_query
        )

        if not check_token_limit(prompt, self.config):
            error_message = "Prompt exceeds token limit"
            logging.error(error_message)
            return None

        # Use the LLM Plugin to get the agent selection
        # Note: LLMPlugin methods are currently synchronous
        llm_result: Optional[str] = self.llm_plugin.generate_text(prompt)

        if llm_result:
            try:
                # Parse the LLM output as JSON
                json_output: Dict[str, Any] = json.loads(llm_result)
                relevant_agents: List[str] = json_output.get("agents", [])
                logging.info(f"LLM-selected agents: {relevant_agents}")
                return relevant_agents
            except json.JSONDecodeError as e:
                logging.error(f"LLM output is not valid JSON: {e}\nOutput: {llm_result}")
                return None  # Or retry, or handle the error in a more sophisticated way
        else:
            logging.error("LLM failed to provide a result.")
            return None  # LLM call failed

    async def run_semantic_kernel_skill(self, skill_name: str, input_vars: Dict[str, str]) -> str:
        """
        Executes a Semantic Kernel skill. This assumes the agent has access to
        a Semantic Kernel instance (self.kernel).
        """
        if not hasattr(self, 'kernel') or not self.kernel:
            raise AttributeError("Agent does not have access to a Semantic Kernel instance (self.kernel)")

        # Get the Semantic Kernel function for the given skill name.
        # This assumes skills are organized in a way that the skill name is also the function name.
        # You might need to adjust this based on how your skills are registered.
        sk_function = self.kernel.functions.get_function(skill_name)

        if not sk_function:
            raise ValueError(f"Semantic Kernel skill '{skill_name}' not found.")

        # Create the Semantic Kernel context.
        sk_context = self.kernel.create_new_context()

        # Set input variables for the Semantic Kernel function.
        for var_name, var_value in input_vars.items():
            sk_context[var_name] = var_value

        # Execute the Semantic Kernel function.
        result = await self.kernel.run(sk_function, input_vars=input_vars)

        # Return the result as a string.
        return str(result)

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the agent's skills (MCP).
        """
        return {
            "name": type(self).__name__,
            "description": self.config.get("description", "Understands user queries and routes them to appropriate agents."),
            "skills": [
                {
                    "name": "understand_query",
                    "description": "Analyzes the user query and determines the relevant agents.",
                    "inputs": [
                        {"name": "user_query", "type": "string", "description": "The user's input query."}
                    ],
                    "outputs": [
                        {"name": "relevant_agents", "type": "list[string]", "description": "A list of agent names."}
                    ]
                }
            ]
        }

# core/agents/query_understanding_agent.py

from core.agents.agent_base import AgentBase  # Assuming you have a base class for agents
from core.utils.config_utils import load_config
from core.utils.token_utils import count_tokens, check_token_limit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QueryUnderstandingAgent(AgentBase):
    """
    An agent responsible for understanding the user's query and
    determining which other agents are relevant to answer it.
    """

    def __init__(self):
        """
        Initializes the QueryUnderstandingAgent.
        """
        super().__init__()  # Initialize the base class
        self.config = load_config("config/agents.yaml")
        agent_config = self.config.get("QueryUnderstandingAgent", {})  # Load specific config
        self.persona = agent_config.get("persona", "A helpful assistant specializing in task delegation.")
        self.expertise = agent_config.get("expertise", ["query analysis", "agent selection"])
        self.prompt_template = agent_config.get("prompt_template",
            "Based on the following user query, identify the most relevant agents to handle the request. "
            "Return a list of agent names.  If no agents are relevant, return an empty list. "
            "Available Agents: {available_agents}\n\nUser Query: {user_query}\n\nRelevant Agents:"
        )
        # "Available Agents" will be a dynamically generated string
        self.available_agents = self.get_available_agents()  # Get a list of available agents


    def get_available_agents(self) -> str:
        """
        Gets a comma-separated string of available agent names from the config.
        Excludes the QueryUnderstandingAgent itself.
        """
        all_agents = self.config.keys()
        # Exclude QueryUnderstandingAgent and any non-agent config entries
        available_agents = [
            agent for agent in all_agents if agent != "QueryUnderstandingAgent" and isinstance(self.config[agent], dict)
        ]
        return ", ".join(available_agents)


    def execute(self, user_query: str) -> list[str]:
        """
        Executes the query understanding process.

        Args:
            user_query: The user's input query.

        Returns:
            A list of agent names (strings) that are deemed relevant to the query.
            Returns an empty list if no agents are relevant.
        """

        if not check_token_limit(user_query, self.config):
            error_message = "User query exceeds the token limit."
            logging.error(error_message)
            return []  # Return an empty list to indicate no agents should be called

        # Construct the prompt
        prompt = self.prompt_template.format(
            available_agents=self.available_agents,
            user_query=user_query
        )

        if not check_token_limit(prompt, self.config):
            error_message = "Prompt exceeds token limit"
            logging.error(error_message)
            return []

        # In a real scenario, you would send the prompt to an LLM here.
        # For this simplified example, we'll use a rule-based approach.
        relevant_agents = self.simple_rule_based_selection(user_query)
        return relevant_agents


    def simple_rule_based_selection(self, user_query: str) -> list[str]:
        """
        A simplified, rule-based agent selection mechanism (placeholder for LLM call).

        This function uses simple keyword matching to determine relevant agents.
        This is a *temporary* solution to be replaced by an LLM call.

        Args:
            user_query: The user's input query.

        Returns:
            A list of agent names.
        """
        relevant_agents = []
        user_query_lower = user_query.lower()

        # Example rules (expand these based on your agents and their capabilities)
        if "risk" in user_query_lower or "credit rating" in user_query_lower:
            relevant_agents.append("RiskAssessmentAgent")
        if "data" in user_query_lower or "retrieve" in user_query_lower:
            relevant_agents.append("DataRetrievalAgent") # Assuming this agent exists
        if "aggregate" in user_query_lower or "combine" in user_query_lower:
            relevant_agents.append("ResultAggregationAgent") #This agent will exist.
        # Add more rules as needed...

        return list(set(relevant_agents))  # Remove duplicates

# Example usage
if __name__ == "__main__":

    # Create a dummy config file (agents.yaml)
    dummy_config = {
    "QueryUnderstandingAgent": {
        "persona": "A query understanding expert.",
        "expertise": ["query analysis", "agent selection"],
        "prompt_template": "Analyze the following query and determine which agents are best suited to handle it. Return a comma-separated list of agent names: {user_query}"
        },
    "RiskAssessmentAgent": {},
    "DataRetrievalAgent": {},
    "ResultAggregationAgent":{}
    }

    with open("config/agents.yaml", "w") as f:
        yaml.dump(dummy_config, f)

    agent = QueryUnderstandingAgent()
    query1 = "What is the credit risk of company XYZ?"
    relevant_agents1 = agent.execute(query1)
    print(f"Query: {query1}\nRelevant Agents: {relevant_agents1}")

    query2 = "Retrieve data on market trends."
    relevant_agents2 = agent.execute(query2)
    print(f"Query: {query2}\nRelevant Agents: {relevant_agents2}")

    query3 = "Aggregate the results."
    relevant_agents3 = agent.execute(query3)
    print(f"Query: {query3}\nRelevant Agents: {relevant_agents3}")

    query4 = "This is a very long query that will hopefully exceed the token limit." * 50
    relevant_agents4 = agent.execute(query4)
    print(f"Query: {query4}\nRelevant Agents: {relevant_agents4}")  # Should return an empty list

    # cleanup
    os.remove("config/agents.yaml")

# core/agents/query_understanding_agent.py

from core.agents.agent_base import AgentBase  # Assuming you have a base class for agents
from core.utils.config_utils import load_config
from core.utils.token_utils import count_tokens, check_token_limit
import logging
from typing import Any, Dict, List, Optional

import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QueryUnderstandingAgent(AgentBase):
    """
    An agent responsible for understanding the user's query and
    determining which other agents are relevant to answer it.
    This version incorporates LLM-based intent recognition and skill-based routing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the QueryUnderstandingAgent.
        """
        super().__init__(config)  # Initialize the base class
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
            logging.warning(
                "QueryUnderstandingAgent: Orchestrator not accessible. Cannot retrieve agent skills."
            )
            return "N/A"  # Or raise an exception, depending on your error handling

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
        llm_result: Optional[str] = await self.llm_plugin.get_completion(prompt)

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


# Example usage (Illustrative - requires a running orchestrator)
# if __name__ == "__main__":
#     # Assuming you have a way to access the orchestrator instance
#     # For example, if it's stored globally or passed as a dependency
#     orchestrator = ... # Get the orchestrator instance
#     agent = QueryUnderstandingAgent(config={}, orchestrator=orchestrator)
#     query1 = "Get the latest market sentiment and analyze the risk for tech companies."
#     relevant_agents1 = asyncio.run(agent.execute(query1))
#     print(f"Query: {query1}\nRelevant Agents: {relevant_agents1}")

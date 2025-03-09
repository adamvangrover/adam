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

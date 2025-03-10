# core/system/interaction_loop.py

from core.system.agent_orchestrator import AgentOrchestrator
from core.system.echo import Echo  # Assuming Echo is used for final output
from core.utils.config_utils import load_config  # For loading configurations
from core.utils.token_utils import check_token_limit, count_tokens  # Import token utilities
import logging  # Import the logging module

# Configure logging (consider moving this to a central location in a real app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class InteractionLoop:
    """
    Handles the main interaction loop of the Adam system.

    This class manages the flow of user input, agent selection,
    data retrieval, result aggregation, and output presentation.
    """

    def __init__(self):
        """
        Initializes the InteractionLoop.
        """
        self.config = load_config("config/system.yaml")  # Load system configuration
        self.agent_orchestrator = AgentOrchestrator()
        self.echo = Echo()  # Initialize the Echo system (placeholder functionality)
        self.token_limit = self.config.get("token_limit", 4096)  # Default to 4096 if not specified
        self.knowledge_base = KnowledgeBase() #instantiate knowledgebase        
        logging.info(f"InteractionLoop initialized with token limit: {self.token_limit}")


    def process_input(self, user_input: str) -> str:
        """
        Processes a single user input and generates a response.

        Args:
            user_input: The user's input string.

        Returns:
            The system's response as a string.
        """
        logging.info(f"Processing input: {user_input}")

        # 1. Token Limit Check (with a small margin)
        if not check_token_limit(user_input, margin=50):  # Allow some tokens for the system's response
            error_message = f"Input exceeds token limit of {self.token_limit - 50}"
            logging.error(error_message)
            return error_message

        # 2. Query Understanding (using QueryUnderstandingAgent)
        query_agent_name = "QueryUnderstandingAgent"  # Hardcoded for now
        query_agent = self.agent_orchestrator.get_agent(query_agent_name)
        if query_agent is None:
            error_message = f"QueryUnderstandingAgent not found."
            logging.error(error_message)
            return error_message

        try:
             # Check and log token count before calling the agent
            input_tokens = count_tokens(user_input)
            logging.info(f"Input token count: {input_tokens}")

            agent_names = query_agent.execute(user_input)

             # Log the response and token count after calling the agent
            response_tokens = count_tokens(str(agent_names))
            logging.info(f"QueryUnderstandingAgent response: {agent_names}, Token count: {response_tokens}")


        except Exception as e:
            error_message = f"Error in QueryUnderstandingAgent: {e}"
            logging.exception(error_message)  # Log the full traceback
            return error_message

        if not isinstance(agent_names, list):
            error_message = "QueryUnderstandingAgent did not return a list of agent names."
            logging.error(error_message)
            return error_message


        # 3. Agent Execution Loop
        results = []
        for agent_name in agent_names:
            agent = self.agent_orchestrator.get_agent(agent_name)
            if agent is None:
                logging.warning(f"Agent '{agent_name}' not found, skipping.")
                continue

            try:
                # Pass necessary context/data to the agent (if needed).  For this simple
                # example, we're passing the original user_input, but in a real
                # scenario, you might pass processed data or specific parameters.
                agent_input = user_input  # Or more specific data
                agent_result = agent.execute(agent_input)

                # Log the agent's response and its token count
                result_tokens = count_tokens(str(agent_result))
                logging.info(f"Agent '{agent_name}' response: {agent_result}, Token count: {result_tokens}")


                results.append(agent_result)
            except Exception as e:
                logging.exception(f"Error executing agent '{agent_name}': {e}")
                # Decide how to handle agent failures.  Here, we continue to the next agent.
                continue

        # 4. Result Aggregation (using ResultAggregationAgent)
        aggregation_agent_name = "ResultAggregationAgent"  # Hardcoded
        aggregation_agent = self.agent_orchestrator.get_agent(aggregation_agent_name)
        if aggregation_agent is None:
            error_message = f"ResultAggregationAgent not found."
            logging.error(error_message)
            return error_message
        try:
            final_result = aggregation_agent.execute(results)

            # Log the final result and its token count
            final_result_tokens = count_tokens(str(final_result))
            logging.info(f"Final aggregated result: {final_result}, Token count: {final_result_tokens}")

        except Exception as e:
            error_message = f"Error in ResultAggregationAgent: {e}"
            logging.exception(error_message)
            return error_message

        # 5. Output (using Echo - for now, just return the string)
        #    In a real application, Echo might format the output, send it to a UI, etc.
        return final_result


    def run(self):
        """
        Starts the continuous interaction loop (not used in this simplified example).

        This method would typically run in a loop, waiting for user input.
        For this example, we'll use the `process_input` method directly.
        """
        print("Adam system interaction loop started.  Type 'exit' to quit.")
        while True:
            user_input = input("Enter your query: ")
            if user_input.lower() == 'exit':
                break
            response = self.process_input(user_input)
            print(response)

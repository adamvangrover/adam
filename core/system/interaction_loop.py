# core/system/interaction_loop.py

import asyncio
import logging  # Import the logging module

from core.system.agent_orchestrator import AgentOrchestrator
from core.system.echo import Echo  # Assuming Echo is used for final output
from core.system.error_handler import AgentNotFoundError, DataNotFoundError, InvalidInputError  # Import exceptions
from core.system.knowledge_base import KnowledgeBase
from core.utils.config_utils import load_config  # For loading configurations
from core.utils.token_utils import check_token_limit, count_tokens  # Import token utilities

# Configure logging (consider moving this to a central location in a real app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class InteractionLoop:
    """
    Handles the main interaction loop of the Adam system.

    This class manages the flow of user input, agent selection,
    data retrieval, result aggregation, and output presentation.
    """

    def __init__(self, config=None, knowledge_base=None):
        """
        Initializes the InteractionLoop.
        """
        self.config = config if config is not None else load_config("config/system.yaml")
        self.agent_orchestrator = AgentOrchestrator()
        self.echo = Echo(self.config)  # Initialize the Echo system (placeholder functionality)
        self.token_limit = self.config.get("token_limit", 4096)  # Default to 4096 if not specified
        self.knowledge_base = knowledge_base if knowledge_base is not None else KnowledgeBase()
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
        if not check_token_limit(user_input, self.config, margin=50):  # Allow some tokens for the system's response
            error_message = f"Input exceeds token limit of {self.token_limit - 50}"
            logging.error(error_message)
            return error_message

        # 2. Query Understanding (using QueryUnderstandingAgent)
        query_agent_name = "QueryUnderstandingAgent"  # Hardcoded for now
        query_agent = self.agent_orchestrator.get_agent(query_agent_name)
        if query_agent is None:
            # RAISE exception instead of return string
            raise AgentNotFoundError(query_agent_name)

        try:
             # Check and log token count before calling the agent
            input_tokens = count_tokens(user_input)
            logging.info(f"Input token count: {input_tokens}")

            # Execute async
            try:
                asyncio.get_running_loop()
                # If loop exists, we can't use run().
                # Ideally we should await, but this method is sync.
                # Assuming this is called from sync context or we need to fail.
                # Use a hack or just fail?
                # For now, let's assume if loop exists, we are in trouble unless we are async.
                # But since we are patching this legacy file, let's try to run it.
                # If loop is running, we can't block.
                raise RuntimeError("Cannot call sync process_input from async loop.")
            except RuntimeError:
                agent_names = asyncio.run(query_agent.execute(user_input))

             # Log the response and token count after calling the agent
            response_tokens = count_tokens(str(agent_names))
            logging.info(f"QueryUnderstandingAgent response: {agent_names}, Token count: {response_tokens}")


        except Exception as e:
            if isinstance(e, (AgentNotFoundError, DataNotFoundError, InvalidInputError)):
                raise
            error_message = f"Error in QueryUnderstandingAgent: {e}"
            logging.exception(error_message)  # Log the full traceback
            # Raise exception if it's an AdamError, otherwise return error message?
            # Test expects InvalidInputError to bubble up
            if hasattr(e, 'code'): # AdamError
                 raise
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
                logging.error(f"Agent '{agent_name}' not found.")
                raise AgentNotFoundError(agent_name)

            try:
                # Pass necessary context/data to the agent (if needed).  For this simple
                # example, we're passing the original user_input, but in a real
                # scenario, you might pass processed data or specific parameters.
                agent_input = user_input  # Or more specific data

                try:
                    asyncio.get_running_loop()
                    raise RuntimeError("Cannot call sync process_input from async loop.")
                except RuntimeError:
                    agent_result = asyncio.run(agent.execute(agent_input))

                # Log the agent's response and its token count
                result_tokens = count_tokens(str(agent_result))
                logging.info(f"Agent '{agent_name}' response: {agent_result}, Token count: {result_tokens}")


                results.append(agent_result)
            except Exception as e:
                if isinstance(e, (AgentNotFoundError, DataNotFoundError, InvalidInputError)):
                    raise
                logging.exception(f"Error executing agent '{agent_name}': {e}")
                # Decide how to handle agent failures.  Here, we continue to the next agent.
                # But if it's a critical error (like DataNotFoundError from the agent), we might want to stop.
                # The test expects DataNotFoundError to bubble up.
                if hasattr(e, 'code'):
                     raise
                continue

        # 4. Result Aggregation (using ResultAggregationAgent)
        aggregation_agent_name = "ResultAggregationAgent"  # Hardcoded
        aggregation_agent = self.agent_orchestrator.get_agent(aggregation_agent_name)
        if aggregation_agent is None:
             raise AgentNotFoundError(aggregation_agent_name)

        try:
            try:
                asyncio.get_running_loop()
                raise RuntimeError("Cannot call sync process_input from async loop.")
            except RuntimeError:
                final_result = asyncio.run(aggregation_agent.execute(results))

            # Log the final result and its token count
            final_result_tokens = count_tokens(str(final_result))
            logging.info(f"Final aggregated result: {final_result}, Token count: {final_result_tokens}")

        except Exception as e:
            if isinstance(e, (AgentNotFoundError, DataNotFoundError, InvalidInputError)):
                raise
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
            try:
                user_input = input("Enter your query: ")
            except EOFError:
                print("\nEOF detected, exiting loop.")
                break
            if user_input.lower() == 'exit':
                break
            try:
                response = self.process_input(user_input)
                print(response)
            except Exception as e:
                print(f"Error: {e}")

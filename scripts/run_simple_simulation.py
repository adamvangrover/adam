# scripts/run_simple_simulation.py
import logging

from core.system.agent_orchestrator import get_orchestrator
from core.system.interaction_loop import InteractionLoop

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_simulation(user_query: str):
    """
    Runs a simple simulation of the Adam system.

    Args:
        user_query: The initial user query.
    """
    try:
        orchestrator = get_orchestrator()
        interaction_loop = InteractionLoop(orchestrator)
        result = interaction_loop.process_input(user_query)
        print(f"Result: {result}")

    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")

if __name__ == "__main__":
    # Hardcoded test query (for the minimal example)
    test_query = "What is the risk rating of company ABC?"
    #test_query = "Give me some market data"
    run_simulation(test_query)

    #You can add additional test queries here, or later create a more complex testing setup.
    # test_query2 = "Send message from AgentA to AgentB: Hello there!"
    # run_simulation(test_query2)

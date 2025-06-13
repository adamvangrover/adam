import sys
import os
import asyncio # Added for asyncio.run
import json # Added for pretty printing skill schema

import logging # Added for WSM or other potential uses

# Assuming core.utils are in the python path or PYTHONPATH is set appropriately
from core.utils.config_utils import load_app_config
from core.utils.logging_utils import setup_logging
from core.utils.prompt_utils import load_system_prompt # Added for system prompt loading
from core.world_simulation.wsm_v7_1 import WorldSimulationModel # Added for WSM integration
from core.agents.echo_agent import EchoAgent # Added for EchoAgent integration
from core.llm_plugin import LLMPlugin # Added for LLMPlugin integration

def main():
    """
    Main entry point for the command-line chatbot.
    """
    try:
        # Load application configuration
        config = load_app_config()

        # Set up logging
        setup_logging(config)

        # Initialize LLMPlugin
        # LLMPlugin loads its own configuration from the specified path.
        # This config (config/llm_plugin.yaml) should point to the mock LLM service.
        try:
            llm_plugin = LLMPlugin(config_path="config/llm_plugin.yaml")
            print("LLMPlugin Initialized (should use mock service as per config/llm_plugin.yaml).")
        except Exception as e:
            print(f"Error initializing LLMPlugin: {e}", file=sys.stderr)
            logging.error(f"LLMPlugin initialization failed: {e}", exc_info=True)
            # Decide if the application should exit or continue without LLM capabilities
            # For this chatbot, LLM might be crucial for some commands.
            sys.exit(1) # Exit if LLMPlugin fails to initialize

        # Load Adam System Prompt
        adam_system_prompt_data = {}
        try:
            system_prompt_filepath = "docs/Adam v19.2 system prompt.txt"
            print(f"Loading system prompt from: {system_prompt_filepath}")
            adam_system_prompt_data = load_system_prompt(system_prompt_filepath)
            if adam_system_prompt_data:
                print("System prompt loaded successfully.")
                # logging.debug(f"System prompt data: {json.dumps(adam_system_prompt_data, indent=2)}")
            else:
                print(f"System prompt file loaded but was empty or invalid JSON content from {system_prompt_filepath}. Proceeding with defaults.")
        except Exception as e:
            print(f"Error loading system prompt: {e}. Proceeding with defaults.")
            logging.error(f"System prompt loading failed: {e}", exc_info=True)

        echo_agent_instructions = adam_system_prompt_data.get('instructions_for_adam', [])
        if not echo_agent_instructions: # Provide a default if not found or empty
            logging.warning("No 'instructions_for_adam' found in system prompt or system prompt failed to load. Using default instructions for EchoAgent.")
            echo_agent_instructions = ["Analyze data thoroughly and provide concise conclusions based on observed patterns."]


        print("Chatbot Initialized. Type 'quit' to exit.")

        while True:
            user_input = input("> ")
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'run_wsm_simulation':
                print("Initializing WSM simulation...")
                try:
                    # Attempt to configure WSM from system prompt - this is illustrative
                    # The prompt structure isn't directly providing WSM params like num_agents in a structured way yet.
                    wsm_params_from_prompt = adam_system_prompt_data.get('world_simulation_model', {})
                    num_agents_from_prompt = wsm_params_from_prompt.get('num_agents', 15) # Default to 15 if not found

                    # For other WSM params like stock_prices, economic_indicators, geopolitical_risks:
                    # These are not clearly defined in the current system_prompt_data structure for direct use in WSM.
                    # We'll use defaults or placeholders for now. Future work could involve defining these more explicitly in the system prompt.
                    default_stock_prices = {
                        'PROMPT_AAPL': [170.0], 'PROMPT_MSFT': [290.0], 'PROMPT_GOOG': [2500.0]
                    }
                    default_econ_indicators = {'sim_gdp_growth': [0.02], 'sim_inflation': [0.03]}
                    default_geo_risks = {'sim_stability_index': [0.75]}

                    wsm_config = {
                        'num_agents': num_agents_from_prompt,
                        'stock_prices': wsm_params_from_prompt.get('stock_prices', default_stock_prices),
                        'economic_indicators': wsm_params_from_prompt.get('economic_indicators', default_econ_indicators),
                        'geopolitical_risks': wsm_params_from_prompt.get('geopolitical_risks', default_geo_risks),
                        'data_sources': wsm_params_from_prompt.get('data_sources', {}) # If WSM uses this
                    }
                    print(f"WSM config (partially from system prompt, with defaults): {wsm_config['num_agents']=}, stocks={list(wsm_config['stock_prices'].keys())}")

                    model = WorldSimulationModel(wsm_config)

                    num_steps = wsm_params_from_prompt.get('simulation_steps', 5) # Default to 5 steps
                    print(f"Running WSM simulation for {num_steps} steps...")
                    for i in range(num_steps):
                        model.step()
                        # Log relevant data from the model if desired
                        # For example, log last stock prices:
                        # current_prices = {k: v[-1] for k, v in model.stock_prices.items()}
                        # logging.info(f"WSM Step {i+1} Stock Prices: {current_prices}")
                        print(f"WSM step {i+1}/{num_steps} completed.")
                    print("WSM simulation finished.")

                    # Extract data from WSM
                    print("Extracting data from WSM...")
                    model_data = model.datacollector.get_model_vars_dataframe()
                    wsm_output_data = model_data

                    print(f"Data extracted. Type: {type(wsm_output_data)}")
                    if wsm_output_data is not None and not wsm_output_data.empty:
                        print("First few rows of WSM model data:")
                        print(wsm_output_data.head())
                    elif wsm_output_data is not None:
                        print("WSM model data is empty.")
                    else:
                        print("WSM model data is None.")

                    # Instantiate EchoAgent
                    print("Initializing EchoAgent for WSM data analysis...")
                    # Pass the main app_config. EchoAgent's __init__ will extract its specific part.
                    echo_agent = EchoAgent(app_config=config,  # Pass the main app_config
                                           llm_plugin_instance=llm_plugin,
                                           adam_instructions=echo_agent_instructions)

                    # Set context for EchoAgent before execution (MCP)
                    current_user_query_objective = "Summarize the key findings from the recent WSM run, considering system guidelines."
                    current_sim_steps = model.schedule.steps if hasattr(model, 'schedule') and hasattr(model.schedule, 'steps') else 'N/A'
                    current_sim_time = model.schedule.time if hasattr(model, 'schedule') and hasattr(model.schedule, 'time') else 'N/A'

                    echo_agent.set_context({
                        "user_query_objective": current_user_query_objective,
                        "simulation_id": f"wsm_run_{current_sim_steps}",
                        "timestamp": current_sim_time,
                        "source_module": __name__
                    })

                    print(f"Invoking EchoAgent to analyze WSM data (simulation ID: wsm_run_{current_sim_steps})...")

                    # Call the async execute method
                    try:
                        # Simplest way to run a single async function from sync code:
                        analysis_result = asyncio.run(echo_agent.execute(wsm_data=wsm_output_data))
                        print(f"EchoAgent Analysis Result: {analysis_result}")
                    except RuntimeError as e_rt:
                        if "asyncio.run() cannot be called from a running event loop" in str(e_rt):
                            # This can happen if run_chatbot.py itself is run within an asyncio loop (e.g. by another framework)
                            # A more sophisticated solution would be needed then, like `await echo_agent.execute(...)`
                            # if main itself was async. For now, log and provide a fallback message.
                            print(f"Asyncio runtime error: {e_rt}. Cannot execute EchoAgent directly. This might happen if run_chatbot is part of a larger async system.")
                            logging.error(f"Asyncio runtime error in run_chatbot: {e_rt}", exc_info=True)
                            analysis_result = "Error: Could not execute EchoAgent due to asyncio conflict."
                        else:
                            raise # Re-raise other runtime errors
                    except Exception as e_async:
                        print(f"Error during EchoAgent async execution: {e_async}")
                        logging.error(f"Error during EchoAgent async execution: {e_async}", exc_info=True)
                        analysis_result = f"Error: {e_async}"


                    # Display skill schema (optional, for verification)
                    # print(f"EchoAgent Skill Schema: {json.dumps(echo_agent.get_skill_schema(), indent=2)}")

                except Exception as wsm_e:
                    print(f"Error during WSM simulation or EchoAgent analysis: {wsm_e}", file=sys.stderr)
                    logging.error(f"WSM simulation or EchoAgent analysis failed: {wsm_e}", exc_info=True)
            else:
                print(f"Chatbot received: {user_input}")

    except Exception as e:
        print(f"Error running Chatbot: {e}", file=sys.stderr)
        logging.error(f"Chatbot main loop failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

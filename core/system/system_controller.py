# core/system/system_controller.py

from plugin_manager import PluginManager
from agents import * # Import all agents
from core.utils.config_utils import load_app_config # Added import

class SystemController:
    def __init__(self, config: dict):
        """
        Initializes the SystemController with a pre-loaded configuration dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        self.plugin_manager = PluginManager(self.config)

        # Initialize agents based on configuration
        # Ensure 'agents' key exists, provide default if necessary
        agents_config = self.config.get("agents", {})
        self.agents = {}
        for agent_name, agent_config in agents_config.items():
            # Ensure agent_config is a dictionary and 'enabled' key exists
            if isinstance(agent_config, dict) and agent_config.get("enabled"):
                agent_class = globals().get(agent_name)  # Use .get for safety
                if agent_class:
                    self.agents[agent_name] = agent_class(agent_config)
                else:
                    # Optionally log a warning if an agent class is not found
                    print(f"Warning: Agent class '{agent_name}' not found.")


    def run(self):
        """
        Runs the main loop of the system, handling user input and agent interactions.
        """
        while True:
            # Get user input
            user_input = input("> ")

            # Process user input and dispatch to appropriate agent or simulation
            if user_input.startswith("!"):
                self.process_command(user_input)
            else:
                pass
                # ... handle other types of input

    def process_command(self, command):
        """
        Processes a user command and dispatches it to the appropriate agent or simulation.

        Args:
            command (str): The user command.
        """
        # Parse command and arguments
        # ...

        # Dispatch command to agent or simulation
        if command_name == "sentiment":
            self.agents["market_sentiment_agent"].analyze_sentiment(command_args)
        elif command_name == "fundamental":
            self.agents["fundamental_analysis_agent"].analyze_company(command_args)
        # ... other commands

    def run_simulation(self, simulation_name, simulation_args):
        """
        Runs the specified simulation with the given arguments.

        Args:
            simulation_name (str): The name of the simulation.
            simulation_args (dict): Arguments for the simulation.
        """
        # Get simulation configuration
        simulation_config = self.config["simulations"][simulation_name]

        # Initialize simulation agents
        simulation_agents = {}
        for agent_name in simulation_config["default_agents"]:
            simulation_agents[agent_name] = self.agents[agent_name]

        # Run the simulation
        # ...

# Example usage
if __name__ == "__main__":
    app_config = load_app_config()
    # Check if 'agents' key is present, as SystemController __init__ expects it.
    # load_app_config should provide it if agents.yaml is loaded.
    if "agents" not in app_config:
        print("Warning: 'agents' key not found in loaded app_config. SystemController might not initialize agents correctly.")
        app_config["agents"] = {} # Provide a default empty dict for agents

    controller = SystemController(config=app_config)
    controller.run()

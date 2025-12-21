import os
import sys


def create_agent(agent_name):
    """
    Creates a new agent file in the core/agents directory.

    Args:
        agent_name (str): The name of the agent to create.
    """
    agent_file_path = f"core/agents/{agent_name.lower()}_agent.py"
    if os.path.exists(agent_file_path):
        print(f"Agent file already exists: {agent_file_path}")
        return

    with open(agent_file_path, "w") as f:
        f.write(f"""from core.agents.agent_base import Agent

class {agent_name}Agent(Agent):
    def __init__(self):
        super().__init__("{agent_name}Agent", "A new agent.")

    def run(self):
        self.log("{agent_name}Agent is running.")
""")

    print(f"Agent file created successfully: {agent_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_agent.py <agent_name>")
        sys.exit(1)
    agent_name = sys.argv[1]
    create_agent(agent_name)

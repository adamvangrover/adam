# core/agents/agent_base.py

class AgentBase:
    def __init__(self):
        pass # Add any common initialization logic for all agents here

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the execute method.")

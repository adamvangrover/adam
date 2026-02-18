from core.agents.agent_base import AgentBase

class {{agent_class_name}}(AgentBase):
    """
    {{agent_description}}
    """
    def __init__(self, config):
        super().__init__(config)
        self.role = "{{agent_role}}"

    async def execute(self, task):
        """
        Executes the agent's main logic.
        """
        print(f"[{self.role}] Executing task: {task}")
        # Add logic here
        return f"Task '{task}' completed by {self.role}."

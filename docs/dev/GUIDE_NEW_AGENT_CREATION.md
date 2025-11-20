Developer Guide: Creating New Agents for Adam v23This guide outlines the standard process for creating, registering, and deploying a new agent within the Adam v23 architecture.PrerequisitesEnsure your environment is set up (see docs/setup_guide.md).Familiarize yourself with core/agents/agent_base.py and core/system/v22_async/async_agent_base.py.Step 1: Define the Agent ConfigurationBefore writing code, define the agent's persona and capabilities in config/agents.yaml (or agents21.yaml depending on your versioning strategy).new_specialist_agent:
  name: "New Specialist Agent"
  role: "specialist"
  description: "An agent dedicated to analyzing [Specific Domain] data."
  model: "gpt-4-turbo" # or configured default
  temperature: 0.3
  tools:
    - "web_search"
    - "internal_data_retrieval"
  system_prompt_path: "prompts/agents/new_specialist.md"
Step 2: Create the System PromptCreate a markdown file at the path specified above (prompts/agents/new_specialist.md).# ROLE
You are the New Specialist Agent. Your primary responsibility is...

# CONSTRAINTS
- Always cite sources.
- Output format must be valid JSON.
- Do not speculate beyond the data provided.

# CAPABILITIES
- Analysis of X
- Synthesis of Y
Step 3: Implement the Agent ClassCreate a new Python file in core/agents/ (e.g., core/agents/new_specialist_agent.py).Use the v23 Template (core/agents/templates/v23_template_agent.py) as your starting point.Key Implementation Details:Inheritance: Must inherit from AsyncAgentBase.Initialization: Call super().__init__().execute_task: This is the main entry point. It must be async.Tool Usage: Use self.tool_manager.execute_tool(). Do not hardcode API calls inside the agent logic if a tool exists.Step 4: Register the AgentYou need to make the system aware of the new class.core/agents/__init__.py: Import your new class.from .new_specialist_agent import NewSpecialistAgent
core/system/agent_orchestrator.py (or agent_factory.py): Add the mapping logic so the orchestrator knows which class to instantiate for the config key new_specialist_agent.Step 5: Integration with Knowledge Graph (v23)If your agent needs to interact with the Unified Knowledge Graph:Ensure self.kg is initialized in your agent.When the agent produces a significant insight, write it back to the graph:await self.kg.add_node(
    label="Insight",
    properties={"content": result, "source": self.name}
)
Step 6: TestingUnit Test: Create tests/test_new_specialist_agent.py. Mock the LLM and Tool Manager responses to verify logic flow.Integration Test: Use scripts/test_new_agents_isolated.py to run the agent with a real (or mocked) LLM against a specific prompt.python scripts/test_new_agents_isolated.py --agent new_specialist_agent --task "Analyze the latest trends in X"
Checklist[ ] Configuration added to YAML.[ ] System Prompt created.[ ] Class implemented using AsyncAgentBase.[ ] Registered in __init__.py.[ ] Unit tests passed.[ ] Documentation updated in core/agents/AGENT_CATALOG.md.

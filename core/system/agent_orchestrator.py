# core/system/agent_orchestrator.py

import logging
import yaml
import os
import importlib
from pathlib import Path
from collections import deque
from typing import Dict, Optional, List, Any

import asyncio  # expand asynchronous communication
import json

from core.agents.agent_base import AgentBase
from core.llm_plugin import LLMPlugin

from core.agents.query_understanding_agent import QueryUnderstandingAgent
from core.agents.data_retrieval_agent import DataRetrievalAgent
from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.macroeconomic_analysis_agent import MacroeconomicAnalysisAgent
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.industry_specialist_agent import IndustrySpecialistAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.technical_analyst_agent import TechnicalAnalystAgent
from core.agents.risk_assessment_agent import RiskAssessmentAgent
from core.agents.newsletter_layout_specialist_agent import NewsletterLayoutSpecialistAgent
from core.agents.data_verification_agent import DataVerificationAgent
from core.agents.lexica_agent import LexicaAgent
from core.agents.archive_manager_agent import ArchiveManagerAgent
from core.agents.agent_forge import AgentForge
from core.agents.prompt_tuner import PromptTuner
from core.agents.code_alchemist import CodeAlchemist
from core.agents.lingua_maestro import LinguaMaestro
from core.agents.sense_weaver import SenseWeaver

from core.utils.config_utils import load_config


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Dictionary mapping agent names to their module paths for dynamic loading
AGENT_CLASSES = {
    "MarketSentimentAgent": "core.agents.market_sentiment_agent",
    "MacroeconomicAnalysisAgent": "core.agents.macroeconomic_analysis_agent",
    "GeopoliticalRiskAgent": "core.agents.geopolitical_risk_agent",
    "IndustrySpecialistAgent": "core.agents.industry_specialist_agent",
    "FundamentalAnalystAgent": "core.agents.fundamental_analyst_agent",
    "TechnicalAnalystAgent": "core.agents.technical_analyst_agent",
    "RiskAssessmentAgent": "core.agents.risk_assessment_agent",
    "NewsletterLayoutSpecialistAgent": "core.agents.newsletter_layout_specialist_agent",
    "DataVerificationAgent": "core.agents.data_verification_agent",
    "LexicaAgent": "core.agents.lexica_agent",
    "ArchiveManagerAgent": "core.agents.archive_manager_agent",
    "AgentForge": "core.agents.agent_forge",
    "PromptTuner": "core.agents.prompt_tuner",
    "CodeAlchemist": "core.agents.code_alchemist",
    "LinguaMaestro": "core.agents.lingua_maestro",
    "SenseWeaver": "core.agents.sense_weaver",
    "QueryUnderstandingAgent": "core.agents.query_understanding_agent",
    "DataRetrievalAgent": "core.agents.data_retrieval_agent",
    "ResultAggregationAgent": "core.agents.result_aggregation_agent",
}


class AgentOrchestrator:
    """
    Manages the creation, execution, and communication of agents.
    This version incorporates MCP and A2A, and leverages Semantic Kernel.
    """

    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.config = load_config("config/agents.yaml")
        self.workflows = self.load_workflows()  # Load workflows
        self.llm_plugin = LLMPlugin()
        self.mcp_service_registry: Dict[
            str, Dict[str, Any]
        ] = {}  # MCP service registry
        if self.config is None:
            logging.error("Failed to load agent configurations.")
        else:
            self.load_agents()
            self.establish_a2a_connections()
            self.register_agent_skills()  # Register agent skills with MCP


    def load_agents(self):
        """Loads agents based on the configuration."""

        if not self.config:
            return

        for agent_name, agent_config in self.config.items():
            if agent_name == "_defaults":  # skip if defaults
                continue
            try:
                agent_class = self._get_agent_class(agent_name)
                if agent_class:
                    self.agents[agent_name] = agent_class(
                        agent_config
                    )  # Pass config to agent
                    logging.info(f"Agent loaded: {agent_name}")
                else:
                    logging.warning(f"Agent class not found for: {agent_name}")
            except Exception as e:
                logging.error(f"Failed to load agent {agent_name}: {e}")

    def _get_agent_class(self, agent_name: str):
        """Retrieves the agent class based on its name."""

        return AGENT_CLASSES.get(agent_name)

    def get_agent(self, agent_name: str) -> Optional[AgentBase]:
        """Retrieves an agent by name."""

        return self.agents.get(agent_name)

    def execute_agent(
        self, agent_name: str, context: Dict[str, Any]
    ) -> Optional[Any]:  # Changed return type to Any
        """Executes an agent with the given context (MCP)."""

        agent = self.get_agent(agent_name)
        if agent:
            try:
                agent.set_context(context)  # Set the MCP context
                return agent.execute(**context)  # Pass context as kwargs
            except Exception as e:
                logging.exception(f"Error executing agent {agent_name}: {e}")
                return None  # Or raise, depending on error handling policy
        else:
            logging.error(f"Agent not found: {agent_name}")
            return None

    async def execute_workflow(
        self, workflow_name: str, initial_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a workflow defined in config/workflows.yaml, handling
        agent dependencies, A2A communication, and MCP skill discovery.
        """

        workflow = self.workflows.get(workflow_name)
        if not workflow:
            logging.error(f"Unknown workflow: {workflow_name}")
            return None

        results: Dict[str, Any] = {}
        execution_queue: deque[str] = deque(workflow["agents"])
        completed_agents: set[str] = set()

        while execution_queue:
            agent_name: str = execution_queue.popleft()
            dependencies: List[str] = workflow["dependencies"].get(agent_name, [])

            if all(dep in completed_agents for dep in dependencies):
                try:
                    agent: Optional[AgentBase] = self.get_agent(agent_name)
                    if not agent:
                        logging.error(
                            f"Agent '{agent_name}' not found for workflow '{workflow_name}'"
                        )
                        return None

                    # Prepare the context for the agent
                    agent_context: Dict[str, Any] = self.prepare_agent_context(
                        agent_name, initial_context, results
                    )

                    # Execute the agent and store the result
                    result: Any = await agent.execute(**agent_context)  # Await agent execution
                    results[agent_name] = result

                    # Handle A2A communication (if any)
                    await self.handle_a2a_communication(
                        agent, agent_name, workflow_name, results, initial_context
                    )

                except Exception as e:
                    logging.error(
                        f"Error executing '{agent_name}' in workflow '{workflow_name}': {e}"
                    )
                    return None

            else:
                execution_queue.append(agent_name)  # Re-queue if dependencies not met

            # Prevent infinite loops
            if agent_name == execution_queue[0] and len(execution_queue) > 1:
                unmet_dependencies: List[str] = [
                    dep for dep in dependencies if dep not in completed_agents
                ]
                logging.error(
                    f"Workflow '{workflow_name}' stuck. Agent '{agent_name}' cannot execute due to unmet dependencies: {unmet_dependencies}"
                )
                return None

        return results  # Return the collected results

    def prepare_agent_context(
        self, agent_name: str, initial_context: Dict[str, Any], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepares the context for an agent before execution.
        This may involve selecting data from the initial context, results
        from previous agents, or data from Semantic Kernel.
        """

        context: Dict[str, Any] = {}
        context.update(initial_context)
        context.update(results)

        # Example: Integrate Semantic Kernel memory
        # if agent_name == "DataRetrievalAgent":
        #     context["semantic_memory"] = self.llm_plugin.get_relevant_memory(initial_context["user_query"])

        return context

    async def handle_a2a_communication(
        self,
        agent: AgentBase,
        agent_name: str,
        workflow_name: str,
        results: Dict[str, Any],
        initial_context: Dict[str, Any],
    ) -> None:
        """
        Handles Agent-to-Agent (A2A) communication during workflow execution.
        This version incorporates more sophisticated A2A logic, including skill-based
        routing and asynchronous message handling.
        """

        # Example: QueryUnderstandingAgent delegates to DataRetrievalAgent
        if agent_name == "QueryUnderstandingAgent" and "user_query" in initial_context:
            user_query: str = initial_context["user_query"]
            if "data" in user_query:
                data_retrieval_agent: Optional[AgentBase] = self.get_agent(
                    "DataRetrievalAgent"
                )
                if data_retrieval_agent:
                    # Prepare A2A message
                    a2a_message: Dict[str, Any] = {"query": user_query}
                    # Send A2A message and await response
                    data: Any = await agent.send_message(
                        "DataRetrievalAgent", a2a_message
                    )  # Use send_message
                    results["DataRetrievalAgent"] = data
                else:
                    logging.warning(
                        f"DataRetrievalAgent not available for A2A in workflow '{workflow_name}'"
                    )

        # Add more sophisticated A2A logic here based on your workflow needs
        # For example, agents might need to exchange intermediate results,
        # request specific information from each other, or negotiate parameters.

        # Placeholder for asynchronous A2A communication
        # Example: Using asyncio.gather to handle multiple A2A messages concurrently
        # a2a_tasks: List[asyncio.Task[Any]] = []
        # for target_agent, message in a2a_messages:
        #    a2a_tasks.append(asyncio.create_task(agent.send_message(target_agent, message)))
        # await asyncio.gather(*a2a_tasks)

        pass

    def load_workflows(self) -> Dict[str, Any]:
        """Loads workflows from config/workflows.yaml."""

        workflows_path: Path = Path("config/workflows.yaml")
        try:
            if workflows_path.exists():
                with workflows_path.open("r") as f:
                    return yaml.safe_load(f)
            else:
                logging.warning(
                    "Workflows configuration file (config/workflows.yaml) not found.  Workflows will not be available."
                )
                return {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing workflows.yaml: {e}")
            return {}
        except Exception as e:
            logging.error(f"Unexpected error loading workflows: {e}")
            return {}

    def run_analysis(self, analysis_type: str, **kwargs: Any) -> Dict[str, Any]:
        """Runs specific analysis tasks using agents."""

        analysis_agents: Dict[str, str] = {
            "market_sentiment": "MarketSentimentAgent",
            "macroeconomic": "MacroeconomicAnalysisAgent",
            "geopolitical_risk": "GeopoliticalRiskAgent",
            "industry_specific": "IndustrySpecialistAgent",
            "fundamental": "FundamentalAnalystAgent",
            "technical": "TechnicalAnalystAgent",
            "risk_assessment": "RiskAssessmentAgent",
        }

        agent_name: Optional[str] = analysis_agents.get(analysis_type)
        if not agent_name:
            return {"error": "Invalid analysis type."}

        return self.execute_agent(agent_name, context=kwargs)  # Pass kwargs as context

    def add_agent(
        self, agent_name: str, agent_type: str, agent_config: Dict[str, Any]
    ) -> None:
        """Dynamically adds and configures a new agent."""

        agent: Optional[AgentBase] = self.create_agent(
            agent_type, agent_config
        )  # Use create_agent
        if agent:
            self.register_agent(agent_name, agent)
            logging.info(f"Added and configured new agent: {agent_name}")
            self.register_agent_skills()  # Update MCP registry
        # else: logging.warning handled in register_agent

    def update_agent_prompt(self, agent_name: str, new_prompt: str) -> None:
        """Updates the prompt template of an agent that supports it."""

        agent: Optional[AgentBase] = self.get_agent(agent_name)
        if agent and hasattr(agent, "prompt_template"):
            agent.prompt_template = new_prompt
            logging.info(f"Updated prompt for agent: {agent_name}")
        else:
            logging.warning(
                f"Agent '{agent_name}' either does not exist or does not support prompt updates."
            )

    def establish_a2a_connections(self) -> None:
        """Establishes Agent-to-Agent (A2A) connections based on config."""

        for agent_name, agent in self.agents.items():
            if hasattr(agent, "peer_agents") and isinstance(
                agent.peer_agents, list
            ):  # Check if peer_agents is defined and is a list
                for peer_agent_name in agent.peer_agents:
                    peer_agent: Optional[AgentBase] = self.get_agent(
                        peer_agent_name
                    )
                    if peer_agent:
                        agent.add_peer_agent(peer_agent)
                        logging.info(
                            f"Established A2A connection: {agent_name} <-> {peer_agent_name}"
                        )
                    else:
                        logging.warning(
                            f"Peer agent '{peer_agent_name}' not found for A2A connection with '{agent_name}'"
                        )

    def register_agent_skills(self) -> None:
        """Registers agent skills in the MCP service registry."""

        self.mcp_service_registry.clear()  # Refresh the registry
        for agent_name, agent in self.agents.items():
            self.mcp_service_registry[agent_name] = agent.get_skill_schema()
            logging.info(f"Registered skills for agent: {agent_name}")

    def discover_agent_skills(self, skill_name: str) -> List[str]:
        """Discovers agents that provide a specific skill (MCP)."""

        providers: List[str] = [
            agent_name
            for agent_name, schema in self.mcp_service_registry.items()
            if any(skill["name"] == skill_name for skill in schema["skills"])
        ]
        return providers

    def route_a2a_message(
        self, target_agent_name: str, message: Dict[str, Any]
    ) -> Optional[Any]:
        """Routes an A2A message to the appropriate agent."""

        target_agent: Optional[AgentBase] = self.get_agent(target_agent_name)
        if target_agent:
            try:
                return target_agent.receive_message(
                    message["sender"], message
                )  # Assuming message contains sender info
            except Exception as e:
                logging.error(
                    f"Error routing A2A message to agent '{target_agent_name}': {e}"
                )
                return None
        else:
            logging.error(f"Target agent '{target_agent_name}' not found for A2A message.")
            return None


def get_orchestrator() -> AgentOrchestrator:
    """Auto-detects runtime and selects the appropriate orchestrator."""

    limited_runtime: bool = os.getenv("LIMITED_RUNTIME", "false").lower() == "true"
    # return SimpleAgentOrchestrator() if limited_runtime else AdvancedAgentOrchestrator()
    return AgentOrchestrator()  # Always use the advanced orchestrator for now


if __name__ == "__main__":
    orchestrator: AgentOrchestrator = get_orchestrator()

    # Create dummy config files for testing purposes
    dummy_agent_config: Dict[str, Any] = {
        "MarketSentimentAgent": {},
        "QueryUnderstandingAgent": {},
        "DataRetrievalAgent": {},  # Add DataRetrievalAgent to the dummy config
    }
    with open("config/agents.yaml", "w") as f:
        yaml.dump(dummy_agent_config, f)
    dummy_workflow_config: Dict[str, Any] = {
        "test_workflow": {
            "agents": ["QueryUnderstandingAgent", "DataRetrievalAgent"],  # Define the workflow
            "dependencies": {},
        }
    }
    with open("config/workflows.yaml", "w") as f:
        yaml.dump(dummy_workflow_config, f)

    # Test basic agent execution
    orchestrator.execute_agent("MarketSentimentAgent", context={})

    # Test workflow execution
    results: Optional[Dict[str, Any]] = None
    if isinstance(orchestrator, AgentOrchestrator):
        results = asyncio.run(
            orchestrator.execute_workflow(
                "test_workflow", initial_context={"user_query": "get data"}
            )
        )
        print(results)

    # --- Example of adding a new agent dynamically ---
    if isinstance(orchestrator, AgentOrchestrator):
        orchestrator.add_agent(
            "MyNewAgent", "MarketSentimentAgent", agent_config={"some_param": "value"}
        )
        orchestrator.execute_agent("MyNewAgent", context={})

    # cleanup files
    os.remove("config/agents.yaml")
    os.remove("config/workflows.yaml")

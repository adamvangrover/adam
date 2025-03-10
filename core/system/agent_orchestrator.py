# core/system/agent_orchestrator.py

import logging
import yaml
import os
import importlib
from pathlib import Path
from collections import deque
from typing import Dict, Optional

import asyncio #expand asynchronous communication // example use asyncio.gather to run the process_query method for agents concurrently
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
    """
    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.config = load_config("config/agents.yaml")
        if self.config is None:
            logging.error("Failed to load agent configurations.")  # This will be caught by the init self-test
        else:
            self.load_agents()


    def load_agents(self):
        """Loads agents based on the configuration."""
        if not self.config:
            return

        for agent_name, agent_config in self.config.items():
            if agent_name == "_defaults": #skip if defaults
                continue
            try:
                agent_class = self._get_agent_class(agent_name)
                self.agents[agent_name] = agent_class(agent_config)  # Pass config to agent
                logging.info(f"Agent loaded: {agent_name}")
            except Exception as e:
                logging.error(f"Failed to load agent {agent_name}: {e}")

    def _get_agent_class(self, agent_name: str):
        """Retrieves the agent class based on its name."""
        agent_classes = {
            "QueryUnderstandingAgent": QueryUnderstandingAgent,
            "DataRetrievalAgent": DataRetrievalAgent,
            "MarketSentimentAgent": MarketSentimentAgent,
            "MacroeconomicAnalysisAgent": MacroeconomicAnalysisAgent,
            "GeopoliticalRiskAgent": GeopoliticalRiskAgent,
            "IndustrySpecialistAgent": IndustrySpecialistAgent,
            "FundamentalAnalystAgent": FundamentalAnalystAgent,
            "TechnicalAnalystAgent": TechnicalAnalystAgent,
            "RiskAssessmentAgent": RiskAssessmentAgent,
            "NewsletterLayoutSpecialistAgent": NewsletterLayoutSpecialistAgent,
            "DataVerificationAgent": DataVerificationAgent,
            "LexicaAgent": LexicaAgent,
            "ArchiveManagerAgent": ArchiveManagerAgent,
            "AgentForge": AgentForge,
            "PromptTuner": PromptTuner,
            "CodeAlchemist": CodeAlchemist,
            "LinguaMaestro": LinguaMaestro,
            "SenseWeaver": SenseWeaver,
        }
        return agent_classes.get(agent_name)

    def get_agent(self, agent_name: str) -> Optional[AgentBase]:
        """Retrieves an agent by name."""
        return self.agents.get(agent_name)

    def execute_agent(self, agent_name: str, *args, **kwargs) -> Optional[str]:
        """Executes an agent with the given arguments."""
        agent = self.get_agent(agent_name)
        if agent:
            try:
                return agent.execute(*args, **kwargs)
            except Exception as e:
                logging.exception(f"Error executing agent {agent_name}: {e}")
                return None
        else:
            logging.error(f"Agent not found: {agent_name}")
            return None

    def execute_workflow(self, workflow_name: str, **kwargs):
        """
        Executes a predefined workflow.  (Placeholder for now)
        Workflows would be defined in config/workflows.yaml.
        """
        # TODO: Implement workflow execution logic.
        logging.warning(f"Workflow execution not yet implemented for: {workflow_name}")
        return None


class BaseAgentOrchestrator:
    """
    Base class for managing agents and executing workflows.
    """

    def __init__(self, config_path="config/agents.yaml"):
        self.agents = {}
        self.config_path = Path(config_path)  # Use Path for file operations
        self.llm_plugin = LLMPlugin()  # Initialize LLM Plugin
        self.load_agents()

    def load_agents(self):
        """Loads agent configurations from a YAML file."""
        try:
            if not self.config_path.exists():
                logging.error(f"Agents configuration file not found: {self.config_path}")
                return

            with self.config_path.open("r") as f:
                agent_configs = yaml.safe_load(f)

            for agent_name, config in agent_configs.items():
                self.register_agent(agent_name, self.create_agent(agent_name, config))

        except yaml.YAMLError as e:
            logging.error(f"Error parsing agents.yaml: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while loading agents: {e}")

    def create_agent(self, agent_name, config):
        """Creates an agent instance based on its name and configuration."""
        try:
            module_name = AGENT_CLASSES.get(agent_name)
            if not module_name:
                logging.warning(f"Agent '{agent_name}' not found in AGENT_CLASSES. Ensure it is registered.")
                return None

            module = importlib.import_module(module_name)
            agent_class = getattr(module, agent_name, None)

            if not agent_class:
                logging.error(f"Agent class '{agent_name}' not found in module '{module_name}'.")
                return None

            return agent_class(config)  # Instantiate the agent with its config

        except (ImportError, AttributeError) as e:
            logging.error(f"Could not load agent '{agent_name}': {e}")
            return None
        except Exception as e:  # Catch any other initialization errors
             logging.error(f"Error initializing agent '{agent_name}': {e}")
             return None

    def register_agent(self, agent_name, agent):
        """Registers an agent."""
        if agent:
            self.agents[agent_name] = agent
            logging.info(f"Agent '{agent_name}' registered successfully.")
        else:
            logging.warning(f"Failed to register agent '{agent_name}'.")

    def get_agent(self, agent_name):
        """Retrieves a registered agent by name."""
        return self.agents.get(agent_name)

    def execute_agent(self, agent_name, *args, **kwargs):
        """Executes a registered agent."""
        agent = self.get_agent(agent_name)
        if agent:
            try:
                return agent.execute(*args, **kwargs)  # Assuming all agents have an execute method
            except Exception as e:
                logging.error(f"Error executing agent '{agent_name}': {e}")
                return None  # Consistent error handling
        else:
            logging.error(f"Agent '{agent_name}' not found.")
            return None


class SimpleAgentOrchestrator(BaseAgentOrchestrator):
    """
    Lightweight orchestrator for constrained environments.
    """

    def __init__(self, config_path="config/agents.yaml"):
        logging.info("Running in lightweight mode (SimpleAgentOrchestrator).")
        super().__init__(config_path)


class AdvancedAgentOrchestrator(BaseAgentOrchestrator):
    """
    Full-featured orchestrator with workflows and dependencies.
    """

    def __init__(self, config_path="config/agents.yaml"):
        logging.info("Running in full mode (AdvancedAgentOrchestrator).")
        super().__init__(config_path)
        self.workflows = self.load_workflows()

    def load_workflows(self):
        """Loads workflows from a YAML file."""
        workflows_path = Path("config/workflows.yaml")  # Use Path
        try:
            if workflows_path.exists():
                with workflows_path.open("r") as f:
                    return yaml.safe_load(f)
            else:
                logging.warning("Workflows configuration file (config/workflows.yaml) not found.  Workflows will not be available.")
                return {}  # Return empty dict if not found
        except yaml.YAMLError as e:
            logging.error(f"Error parsing workflows.yaml: {e}")
            return {} # Return empty dict on parsing error
        except Exception as e:
            logging.error(f"Unexpected error loading workflows: {e}")
            return {}

    def execute_workflow(self, workflow_name, **kwargs):
        """
        Executes a predefined workflow, handling dependencies.
        Uses a queue to manage agent execution order.
        """
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            logging.error(f"Unknown workflow: {workflow_name}")
            return None  # Consistent error return

        results = {}  # Store results from each agent
        execution_queue = deque(workflow["agents"])  # Use a deque for efficient queue operations
        completed_agents = set()

        while execution_queue:
            agent_name = execution_queue.popleft()  # Get the next agent from the queue
            dependencies = workflow["dependencies"].get(agent_name, [])

            if all(dep in completed_agents for dep in dependencies):
                # All dependencies are met, execute the agent
                try:
                    result = self.execute_agent(agent_name, **kwargs)
                    results[agent_name] = result  # Store the result
                    completed_agents.add(agent_name)
                except Exception as e:
                    logging.error(f"Error executing '{agent_name}' in workflow '{workflow_name}': {e}")
                    # Consider whether to halt the entire workflow or continue
                    return None  # Or raise, or continue with a partial result

            else:
                # Dependencies not met, re-add the agent to the end of the queue
                execution_queue.append(agent_name)

                # Prevent infinite loops due to unmet dependencies.  Check if we've
                # gone through the entire queue without executing any agents.
                if agent_name == execution_queue[0] and len(execution_queue) > 1 :
                   unmet_dependencies = [dep for dep in dependencies if dep not in completed_agents]
                   logging.error(f"Workflow '{workflow_name}' stuck. Agent '{agent_name}' cannot execute due to unmet dependencies: {unmet_dependencies}")
                   return None # Exit the workflow

        return results  # Return the collected results



    def run_analysis(self, analysis_type, **kwargs):
        """Runs specific analysis tasks using agents."""
        # This could also be loaded from a config file for greater flexibility
        analysis_agents = {
            "market_sentiment": "MarketSentimentAgent",
            "macroeconomic": "MacroeconomicAnalysisAgent",
            "geopolitical_risk": "GeopoliticalRiskAgent",
            "industry_specific": "IndustrySpecialistAgent",
            "fundamental": "FundamentalAnalystAgent",
            "technical": "TechnicalAnalystAgent",
            "risk_assessment": "RiskAssessmentAgent",
        }

        agent_name = analysis_agents.get(analysis_type)
        if not agent_name:
            return {"error": "Invalid analysis type."}  # Consistent error structure

        return self.execute_agent(agent_name, **kwargs)


    def add_agent(self, agent_name, agent_type, **kwargs):
        """Dynamically adds and configures a new agent."""
        agent_config = kwargs # The kwargs *are* the config
        agent = self.create_agent(agent_type, agent_config) # Use create_agent
        if agent:
            self.register_agent(agent_name, agent)
            logging.info(f"Added and configured new agent: {agent_name}")
        # else: logging.warning handled in register_agent


    def update_agent_prompt(self, agent_name, new_prompt: str):
        """Updates the prompt template of an agent that supports it."""
        agent = self.get_agent(agent_name)
        if agent and hasattr(agent, "prompt_template"):
            # Agents that support dynamic prompts should have a prompt_template attribute.
            agent.prompt_template = new_prompt
            logging.info(f"Updated prompt for agent: {agent_name}")
        else:
            logging.warning(
                f"Agent '{agent_name}' either does not exist or does not support prompt updates."
            )


def get_orchestrator():
    """Auto-detects runtime and selects the appropriate orchestrator."""
    limited_runtime = os.getenv("LIMITED_RUNTIME", "false").lower() == "true"
    return SimpleAgentOrchestrator() if limited_runtime else AdvancedAgentOrchestrator()


if __name__ == "__main__":
    orchestrator = get_orchestrator()

    # Create dummy config files for testing purposes
    dummy_agent_config = {
        "MarketSentimentAgent": {},
        "QueryUnderstandingAgent":{}
    }
    with open("config/agents.yaml", "w") as f:
        yaml.dump(dummy_agent_config, f)
    dummy_workflow_config = {
        "test_workflow": {
            "agents": ["MarketSentimentAgent", "QueryUnderstandingAgent"],
            "dependencies": {}
        }
    }
    with open("config/workflows.yaml", 'w') as f:
        yaml.dump(dummy_workflow_config, f)

    # Test basic agent execution
    orchestrator.execute_agent("MarketSentimentAgent")

    # Test workflow execution (if AdvancedAgentOrchestrator)
    if isinstance(orchestrator, AdvancedAgentOrchestrator):
        results = orchestrator.execute_workflow("test_workflow")
        print(results)

    # --- Example of adding a new agent dynamically ---
    if isinstance(orchestrator, AdvancedAgentOrchestrator):
        orchestrator.add_agent("MyNewAgent", "MarketSentimentAgent", some_param="value")
        orchestrator.execute_agent("MyNewAgent")

    # cleanup files
    os.remove("config/agents.yaml")
    os.remove("config/workflows.yaml")

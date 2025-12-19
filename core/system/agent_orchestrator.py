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
from core.agents.newsletter_layout_specialist_agent import NewsletterLayoutSpecialist as NewsletterLayoutSpecialistAgent
from core.agents.data_verification_agent import DataVerificationAgent
from core.agents.lexica_agent import LexicaAgent
from core.agents.archive_manager_agent import ArchiveManagerAgent
from core.agents.agent_forge import AgentForge
from core.agents.prompt_tuner import PromptTuner
from core.agents.code_alchemist import CodeAlchemist
from core.agents.lingua_maestro import LinguaMaestro
from core.agents.sense_weaver import SenseWeaver
from core.agents.snc_analyst_agent import SNCAnalystAgent # Added import
from core.agents.behavioral_economics_agent import BehavioralEconomicsAgent
from core.agents.meta_cognitive_agent import MetaCognitiveAgent

from pydantic import ValidationError
from core.schemas.config_schema import AgentsYamlConfig

from core.utils.config_utils import load_config
from core.utils.secrets_utils import get_api_key # Added import
from core.system.message_broker import MessageBroker
# from core.system.brokers.rabbitmq_client import RabbitMQClient

# Semantic Kernel imports
try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
except ImportError:
    Kernel = None
    OpenAIChatCompletion = None
    logging.warning("Semantic Kernel not installed. Skipping SK integration.")
# from semantic_kernel.connectors.ai.azure_open_ai import AzureChatCompletion # Example for later
# from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion # Example for later


# Configure logging
# Ensure logging is configured. If it's already configured at a higher level (e.g. main script),
# this line might be redundant or could be adjusted. For now, keeping it.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from financial_digital_twin.nexus_agent import NexusAgent
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
    "ReportGeneratorAgent": "core.agents.report_generator_agent",
    "SNCAnalystAgent": "core.agents.snc_analyst_agent", # Added snc_analyst_agent
    "BehavioralEconomicsAgent": "core.agents.behavioral_economics_agent",
    "MetaCognitiveAgent": "core.agents.meta_cognitive_agent",
    "NewsBotAgent": "core.agents.news_bot",
    "NexusAgent": NexusAgent,
    "IngestionAgent": AgentBase, # Using AgentBase as a placeholder
    "AuditorAgent": AgentBase, # Using AgentBase as a placeholder
}


class AgentOrchestrator:
    """
    Manages the creation, execution, and communication of agents.
    This version incorporates MCP and A2A, and leverages Semantic Kernel.
    """

    def __init__(self):
        self.agents: Dict[str, AgentBase] = {}
        self.config = load_config("config/config.yaml") or {}
        
        # Load agents.yaml if config.yaml is empty or missing agents
        if 'agents' not in self.config:
            agents_cfg = load_config("config/agents.yaml")
            if agents_cfg and 'agents' in agents_cfg:
                try:
                    # Validate configuration against Pydantic schema
                    validated_config = AgentsYamlConfig(**agents_cfg)
                    # Use the validated data
                    self.config['agents'] = validated_config.model_dump(exclude_none=True)['agents']
                    logging.info("Agents configuration validated successfully against strict schema.")
                except ValidationError as e:
                    logging.critical(f"Agent configuration validation failed: {e}")
                    # Fail fast as recommended by analysis
                    raise RuntimeError(f"Startup Aborted: Invalid agent configuration in agents.yaml. \n{e}")
                except Exception as e:
                    logging.exception(f"Unexpected error during config validation: {e}")
                    raise

        self.workflows = self.load_workflows()  # Load workflows
        self.llm_plugin = LLMPlugin()
        self.mcp_service_registry: Dict[
            str, Dict[str, Any]
        ] = {}  # MCP service registry
        self.sk_kernel: Optional[Kernel] = None # Initialize Semantic Kernel instance to None
        # Use In-Memory Broker for default/showcase (RabbitMQ requires external service)
        self.message_broker = MessageBroker.get_instance()
        self.message_broker.connect()

        if self.config is None:
            logging.error("Failed to load agent configurations (config/config.yaml).")
        else:
            self.load_agents()
            self.establish_a2a_connections()
            self.register_agent_skills()  # Register agent skills with MCP

        # Initialize Semantic Kernel
        if Kernel is not None:
            try:
                sk_settings_config = load_config('config/semantic_kernel_settings.yaml')
                if not sk_settings_config or 'semantic_kernel_settings' not in sk_settings_config:
                    logging.error("Semantic Kernel settings not found or empty in config/semantic_kernel_settings.yaml. SK will not be available.")
                    # self.sk_kernel remains None
                else:
                    sk_settings = sk_settings_config['semantic_kernel_settings']
                    default_service_id = sk_settings.get('default_completion_service_id')
                    completion_services = sk_settings.get('completion_services')

                    if not default_service_id or not completion_services:
                        logging.error("Default service ID or completion_services not defined in Semantic Kernel settings. SK will not be available.")
                        # self.sk_kernel remains None
                    else:
                        service_config = completion_services.get(default_service_id)
                        if not service_config:
                            logging.error(f"Configuration for default service ID '{default_service_id}' not found in completion_services. SK will not be available.")
                            # self.sk_kernel remains None
                        else:
                            kernel_instance = Kernel()
                            service_type = service_config.get('service_type')
                            model_id = service_config.get('model_id')

                            if service_type == "OpenAI":
                                api_key = get_api_key('OPENAI_API_KEY')
                                # Optional: Fetch org_id if specified in sk_settings for this service
                                # org_id_env_var = service_config.get('org_id_env_var')
                                # org_id = get_api_key(org_id_env_var) if org_id_env_var else None
                                org_id = get_api_key('OPENAI_ORG_ID') # Simpler: always try OPENAI_ORG_ID

                                if api_key and model_id:
                                    try:
                                        service_instance = OpenAIChatCompletion(
                                            model_id=model_id,
                                            api_key=api_key,
                                            org_id=org_id if org_id else None
                                        )
                                        kernel_instance.add_chat_service("default_completion", service_instance)
                                        self.sk_kernel = kernel_instance # Assign successfully configured kernel
                                        logging.info(f"Semantic Kernel initialized with OpenAI service '{default_service_id}' (model: {model_id}).")
                                    except Exception as e:
                                        logging.error(f"Failed to initialize and add OpenAI service to Semantic Kernel: {e}")
                                        # self.sk_kernel remains None as it was before this block
                                else:
                                    logging.error("OpenAI API key or model_id missing. Cannot configure OpenAI service for Semantic Kernel. SK will not be available.")
                                    # self.sk_kernel remains None
                            # elif service_type == "AzureOpenAI":
                                # Placeholder for Azure OpenAI configuration
                                # logging.info("AzureOpenAI service type found, implementation pending.")
                            # elif service_type == "HuggingFace":
                                # Placeholder for HuggingFace configuration
                                # logging.info("HuggingFace service type found, implementation pending.")
                            else:
                                logging.error(f"Unsupported Semantic Kernel service type: {service_type}. SK will not be available.")
                                # self.sk_kernel remains None
            except Exception as e:
                logging.error(f"An unexpected error occurred during Semantic Kernel initialization: {e}")
                self.sk_kernel = None # Ensure it's None on any exception during init
        else:
            self.sk_kernel = None

        # Load Semantic Kernel skills if kernel was initialized
        if self.sk_kernel:
            skills_directory = "core/agents/skills/"
            if os.path.isdir(skills_directory):
                for skill_collection_name in os.listdir(skills_directory):
                    skill_collection_path = os.path.join(skills_directory, skill_collection_name)
                    if os.path.isdir(skill_collection_path):
                        try:
                            logging.info(f"Importing SK skill collection: {skill_collection_name} from {skill_collection_path}")
                            # For SK v0.x Python: kernel.import_skill(skill_instance, skill_name) or for directory:
                            # kernel.import_semantic_skill_from_directory(parent_directory, skill_directory_name)
                            # or kernel.import_skill_from_directory (newer pre-v1)
                            # The problem asks for import_skill(parent_directory=..., skill_collection_name=...)
                            # which is slightly different from standard.
                            # A common pattern for directory import is:
                            # self.sk_kernel.import_skill(parent_directory=skills_directory, skill_directory=skill_collection_name)
                            # or using plugins: self.sk_kernel.add_plugin(parent_directory=skills_directory, plugin_name=skill_collection_name)
                            # Given the context of "skills" and "collections", and the AgentBase method using
                            # kernel.skills.get_function, this implies a structure where skills are registered under a collection name.
                            # The method import_skill_from_directory(parent_directory, skill_directory_name) is appropriate here.
                            # skill_directory_name becomes the collection name.
                            self.sk_kernel.import_skill_from_directory(skills_directory, skill_collection_name)
                            logging.info(f"Successfully imported SK skill collection: {skill_collection_name}")
                        except Exception as e:
                            logging.error(f"Failed to import SK skill collection '{skill_collection_name}': {e}")
            else:
                logging.warning(f"Skills directory '{skills_directory}' not found. No SK skills loaded.")


    def load_agents(self):
        """Loads agents based on the new configuration structure."""
        if not self.config or 'agents' not in self.config:
            logging.error("Agent configuration 'agents' key not found in config.yaml.")
            return

        agents_data = self.config.get('agents', [])
        
        # Normalize dict to list-like iteration
        iterator = []
        if isinstance(agents_data, dict):
            for name, cfg in agents_data.items():
                if cfg is None: cfg = {}
                cfg['name'] = name # Inject name from key
                iterator.append(cfg)
        elif isinstance(agents_data, list):
            iterator = agents_data
            
        for agent_config in iterator:
            agent_name = agent_config.get('name')
            if not agent_name:
                logging.warning("Skipping agent configuration with no name.")
                continue

            try:
                agent_class = self._get_agent_class(agent_name)
                if agent_class:
                    constitution = None
                    constitution_path = agent_config.get('constitution_path')
                    if constitution_path:
                        try:
                            with open(constitution_path, 'r') as f:
                                constitution = json.load(f)
                        except FileNotFoundError:
                            logging.error(f"Constitution file not found for {agent_name} at {constitution_path}")
                        except json.JSONDecodeError:
                            logging.error(f"Failed to decode JSON from constitution file for {agent_name} at {constitution_path}")

                    # Pass agent_config, constitution, and kernel to the agent
                    self.agents[agent_name] = agent_class(
                        config=agent_config,
                        constitution=constitution,
                        kernel=self.sk_kernel
                    )
                    logging.info(f"Agent loaded: {agent_name}")
                else:
                    logging.warning(f"Agent class not found for: {agent_name}")
            except Exception as e:
                logging.error(f"Failed to load agent {agent_name}: {e}", exc_info=True)

    def _get_agent_class(self, agent_name: str):
        """Retrieves the agent class based on its name."""

        agent_entry = AGENT_CLASSES.get(agent_name)

        # If the entry is already a class type, return it
        if isinstance(agent_entry, type):
            return agent_entry

        # If it's a string, try to resolve it
        if isinstance(agent_entry, str):
            # Check if it's already imported and available in globals
            if agent_name in globals():
                return globals()[agent_name]

            # Fallback to dynamic import
            try:
                module_path = agent_entry
                module = importlib.import_module(module_path)

                # Assume class name matches agent name
                if hasattr(module, agent_name):
                    return getattr(module, agent_name)

            except Exception as e:
                logging.error(f"Failed to dynamically load agent {agent_name}: {e}")

        return None

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
                # Instead of direct execution, publish a message
                # The message should contain the context and a reply-to topic
                reply_topic = f"{agent_name}_results"
                message = {
                    "context": context,
                    "reply_to": reply_topic
                }
                self.message_broker.publish(agent_name, json.dumps(message))

                # For this example, we'll assume a synchronous response
                # In a real-world scenario, you would have a separate process
                # listening for responses.
                # This is a simplified simulation of the async process.
                # We'll need a mechanism to wait for the response.
                # For now, let's just log that the message was published.
                logging.info(f"Published task for {agent_name} to message broker.")
                return None
            except Exception as e:
                logging.exception(f"Error executing agent {agent_name}: {e}")
                return None
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
            logging.info(f"Workflow '{workflow_name}' not found. Attempting dynamic workflow generation.")
            # Attempt to generate a dynamic workflow
            try:
                # Prepare the input for the WorkflowCompositionSkill
                available_skills = json.dumps(self.mcp_service_registry, indent=2)
                user_query = initial_context.get("user_query", "")

                # Run the Semantic Kernel skill
                generated_workflow_str = await self.sk_kernel.run_async(
                    self.sk_kernel.skills.get_function("WorkflowCompositionSkill", "compose"),
                    input_vars={"input": user_query, "skills": available_skills}
                )

                # Parse and validate the generated workflow
                workflow = yaml.safe_load(generated_workflow_str)
                if not workflow or "agents" not in workflow or "dependencies" not in workflow:
                    logging.error("Dynamic workflow generation failed: Invalid workflow format.")
                    return None

                logging.info(f"Dynamically generated workflow: {workflow}")

            except Exception as e:
                logging.error(f"Error during dynamic workflow generation: {e}")
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
            "behavioral_economics": "BehavioralEconomicsAgent",
            "meta_cognitive": "MetaCognitiveAgent",
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
        """Establishes Agent-to-Agent (A2A) connections based on agent configurations."""

        for agent_name, agent in self.agents.items():
            # Get peer agent names from the current agent's configuration
            peer_agent_names = agent.config.get('peers', [])
            
            if isinstance(peer_agent_names, list):
                for peer_name in peer_agent_names:
                    peer_agent_instance = self.get_agent(peer_name)
                    if peer_agent_instance:
                        # The add_peer_agent method in AgentBase should handle the actual adding.
                        # AgentBase.add_peer_agent already logs this connection.
                        # We can rely on that, or add more specific logging here if needed.
                        # For example, using agent.config.get('name', agent_name) for a more descriptive name if available.
                        agent.add_peer_agent(peer_agent_instance)
                        # Logging example (can be redundant if add_peer_agent logs sufficiently):
                        # current_agent_display_name = agent.config.get('persona', agent_name) # Using persona as a display name
                        # peer_agent_display_name = peer_agent_instance.config.get('persona', peer_name)
                        # logging.info(f"A2A connection attempt: {current_agent_display_name} -> {peer_agent_display_name} successful via config.")
                    else:
                        logging.warning(
                            f"Peer agent '{peer_name}' listed in config for '{agent.config.get('persona', agent_name)}' not found."
                        )
            elif peer_agent_names: # If 'peers' exists but is not a list
                 logging.warning(
                    f"Invalid 'peers' configuration for agent '{agent.config.get('persona', agent_name)}': "
                    f"'peers' should be a list, but found type {type(peer_agent_names)}."
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

# core/agents/echo_agent.py

import os
import json
import logging
from core.utils.data_utils import send_message, receive_messages
from core.utils.api_utils import get_knowledge_graph_data
# LLMPlugin import might be needed if type hinting, but not for runtime if instance is passed
# from core.llm_plugin import LLMPlugin # LLMPlugin type hint can be used if available globally
from typing import List, Optional, Dict, Any # For type hinting
from .agent_base import AgentBase # Import AgentBase

# Ensure logging is configured if not already by a higher-level module
# This is a good practice for library-like modules.
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class EchoAgent(AgentBase): # Inherit from AgentBase
    def __init__(self,
                 app_config: Dict[str, Any], # Renamed config to app_config for clarity
                 llm_plugin_instance: Optional[Any] = None,
                 adam_instructions: Optional[List[str]] = None):

        # Extract agent-specific config for super()
        # The original EchoAgent used app_config directly or app_config.get('echo_system', {})
        # We need to ensure 'name' is in the agent_specific_config for AgentBase or its own logger.
        agent_specific_config = app_config.get('agents', {}).get('EchoAgent', {})
        if not agent_specific_config: # Fallback if 'EchoAgent' not under 'agents'
            agent_specific_config = app_config.get('echo_system', app_config) # Use app_config as last resort for compatibility

        if 'name' not in agent_specific_config:
            agent_specific_config['name'] = 'EchoAgent' # Default name

        super().__init__(config=agent_specific_config, kernel=None) # Pass agent_specific_config to AgentBase

        # Initialize EchoAgent-specific attributes
        self.llm_plugin = llm_plugin_instance
        self.adam_instructions = adam_instructions if adam_instructions else []

        # self.logger is initialized by AgentBase using type(self).__name__ if not in config.
        # If a specific name from config is desired for EchoAgent's logger:
        self.logger = logging.getLogger(self.config.get('name', 'EchoAgent')) # self.config is from AgentBase

        # self.config from AgentBase now holds agent_specific_config.
        # Methods like detect_environment should use this self.config.
        self.environment = self.detect_environment() # detect_environment needs to use self.config
        self.ui_enabled = self.config.get('ui_enabled', False)

        self.logger.info(f"EchoAgent '{self.config.get('name')}' fully initialized.")


    def detect_environment(self):
        """
        Detects the environment settings including LLM engine, API type, and system resources.
        """
        try:
            # 1. LLM Engine Identification
            llm_engine = os.environ.get('LLM_ENGINE', 'unknown')  # Example: 'openai', 'google'

            # 2. API Type Determination
            api_type = 'simulated' if 'simulated' in self.config.get('data_sources', {}) else 'actual'

            # 3. Resource Availability Assessment (Example: CPU and memory check)
            resources = {
                'cpu': os.cpu_count(),
                'memory': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)  # in GB
            }

            environment = {
                'llm_engine': llm_engine,
                'api_type': api_type,
                'resources': resources,
            }

            self.logger.info("Environment detected: %s", environment)
            return environment

        except Exception as e:
            self.logger.error("Error detecting environment: %s", e)
            return {}

    def optimize_prompt(self, prompt, **kwargs):
        """
        Optimizes the prompt based on the current environment and task context.
        """
        try:
            optimized_prompt = prompt  # Placeholder for optimization logic
            if self.environment.get('llm_engine') == 'openai':
                # Example optimization: Customize prompt for OpenAI engine
                optimized_prompt = f"Optimized for OpenAI: {prompt}"

            self.logger.info("Prompt optimized: %s", optimized_prompt)
            return optimized_prompt

        except Exception as e:
            self.logger.error("Error optimizing prompt: %s", e)
            return prompt

    def run_ui(self):
        """
        Starts the chatbot UI if enabled.
        """
        try:
            if self.ui_enabled:
                # Placeholder for actual UI implementation
                self.logger.info("Starting chatbot UI...")
                pass  # Start UI code here
            else:
                self.logger.info("UI not enabled.")

        except Exception as e:
            self.logger.error("Error starting UI: %s", e)

    def run_expert_network(self, task, **kwargs):
        """
        Run a network of expert models to process the task.
        """
        try:
            # Placeholder for expert network logic
            self.logger.info("Running expert network for task: %s", task)
            pass  # Implement expert network logic here

        except Exception as e:
            self.logger.error("Error running expert network: %s", e)

    def enhance_output(self, output, **kwargs):
        """
        Enhance the output using additional knowledge or insights.
        """
        try:
            enhanced_output = output  # Placeholder for enhancing logic
            self.logger.info("Output enhanced: %s", enhanced_output)
            return enhanced_output

        except Exception as e:
            self.logger.error("Error enhancing output: %s", e)
            return output

    def get_knowledge_graph_context(self, query):
        """
        Retrieves relevant context from the knowledge graph based on the query.
        """
        try:
            context = get_knowledge_graph_data(query)  # Example function call
            self.logger.info("Knowledge graph context retrieved for query '%s': %s", query, context)
            return context

        except Exception as e:
            self.logger.error("Error fetching knowledge graph context: %s", e)
            return {}

    def process_task(self, ch, method, properties, body):
        """
        Processes the received task from the message queue.
        """
        try:
            task = json.loads(body)
            task_type = task.get('type')
            self.logger.info("Processing task of type: %s", task_type)

            if task_type == 'optimize_prompt':
                optimized_prompt = self.optimize_prompt(task['prompt'], **task.get('kwargs', {}))
                send_message(queue='echo_agent_responses', message=optimized_prompt)

            elif task_type == 'run_expert_network':
                self.run_expert_network(task['task'], **task.get('kwargs', {}))

            elif task_type == 'enhance_output':
                enhanced_output = self.enhance_output(task['output'], **task.get('kwargs', {}))
                send_message(queue='echo_agent_responses', message=enhanced_output)

            elif task_type == 'get_knowledge_graph_context':
                context = self.get_knowledge_graph_context(task['query'])
                send_message(queue='echo_agent_responses', message=context)

            else:
                self.logger.warning("Unknown task type: %s", task_type)

        except Exception as e:
            self.logger.error("Error processing task: %s", e)

    def run(self, **kwargs):
        """
        Main entry point for running the agent.
        """
        try:
            # 1. Receive task from message queue and process it
            receive_messages(queue='echo_agent_tasks', callback=self.process_task)

        except Exception as e:
            self.logger.error("Error running EchoAgent: %s", e)

    def analyze_simulation_data(self, wsm_data):
        """
        Analyzes data received from the World Simulation Model.
        """
        self.logger.info(f"'{self.config.get('name', type(self).__name__)}' received WSM data for analysis.")

        if not self.llm_plugin:
            self.logger.error(f"'{self.config.get('name', type(self).__name__)}': LLMPlugin instance not provided. Cannot analyze WSM data with LLM.")
            return "Error: LLMPlugin not available for analysis."

        if wsm_data is None:
            self.logger.warning("Received WSM data is None.")
            return "WSM data was None, no analysis performed by LLM."

        # Check if wsm_data is a DataFrame and if it's empty
        if hasattr(wsm_data, 'empty') and wsm_data.empty:
            self.logger.warning("Received WSM data is an empty DataFrame.")
            return "WSM data was empty, no analysis performed by LLM."

        try:
            # Convert DataFrame to string for the prompt context
            # For other data types, adjust as needed.
            if hasattr(wsm_data, 'to_string'): # Handles pandas DataFrame
                data_summary_for_prompt = "WSM Data Summary:\n" + wsm_data.to_string()
            else: # Basic fallback for other types (e.g. dict)
                data_summary_for_prompt = "WSM Data Summary:\n" + str(wsm_data)

            # Limit data summary size
            if len(data_summary_for_prompt) > 2500: # Reserve some space for instructions and query
                data_summary_for_prompt = data_summary_for_prompt[:2500] + "\n... (WSM data summary truncated)"

            # Construct a more prominent guidelines section
            guidelines_section = "SYSTEM GUIDELINES FOR ANALYSIS:\n"
            if self.adam_instructions:
                # Select a few key instructions based on keywords or fallback to first few
                selected_instructions = [
                    instr for instr in self.adam_instructions
                    if any(kw in instr.lower() for kw in ["accuracy", "transparency", "explainability", "actionable", "relevance", "timeliness", "chain-of-thought"])
                ]
                if not selected_instructions and len(self.adam_instructions) > 0:
                    selected_instructions = self.adam_instructions[:3] # Fallback to first 3 if no keyword matches
                elif not selected_instructions: # Still no instructions
                     selected_instructions = ["Analyze data thoroughly and provide concise, actionable conclusions."]


                for i, instr in enumerate(selected_instructions[:4]): # Limit to max 4 diverse instructions
                    guidelines_section += f"- {instr}\n"
            else: # Default if no adam_instructions were provided at all
                guidelines_section += "- Analyze data thoroughly and provide concise, actionable conclusions.\n"
                guidelines_section += "- Focus on accuracy, relevance, and timeliness.\n"

            # Combine guidelines and data summary into the context for the LLM
            # Ensure a clear separation for the LLM to distinguish guidelines from data.
            full_context_for_llm = guidelines_section + "\nSIMULATION DATA CONTEXT:\n" + data_summary_for_prompt

            # Refined user_query to be more open-ended yet directive
            user_query = ("Provide a comprehensive analysis of the following simulation data, "
                          "adhering to the system guidelines. Focus on key observations, "
                          "potential implications, and actionable insights where appropriate. "
                          "Employ Chain-of-Thought reasoning if beneficial for clarity.")

            self.logger.info(f"'{self.config.get('name', type(self).__name__)}' sending query to LLMPlugin: {user_query[:200]}...")
            self.logger.debug(f"Full context for LLM (first 1000 chars): {full_context_for_llm[:1000]}...") # Log more context

            # Use the LLMPlugin to get a response
            llm_response = self.llm_plugin.generate_content(
                context=full_context_for_llm,
                prompt_text=user_query
            )
            self.logger.info(f"'{self.config.get('name', type(self).__name__)}' received response from LLMPlugin: {llm_response}")
            return f"Simulated LLM Conclusion: {llm_response}"

        except Exception as e:
            self.logger.error(f"'{self.config.get('name', type(self).__name__)}' error during LLMPlugin interaction or data processing: {e}", exc_info=True)
            return f"Error interacting with LLMPlugin or processing WSM data: {str(e)}"


    # --- MCP Methods Implementation ---

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines EchoAgent's skills according to MCP.
        """
        return {
            "name": self.config.get("name", type(self).__name__),
            "description": self.config.get("description", "Agent that analyzes simulation data using an LLM and system guidelines."),
            "skills": [
                {
                    "name": "analyze_simulation_data", # This is the primary function this agent exposes
                    "description": "Processes data from a World Simulation Model, consults an LLM based on system instructions, and returns conclusions.",
                    "parameters": [
                        {"name": "wsm_data", "type": "DataFrame", "description": "Data output from WSM (Pandas DataFrame expected)."}
                    ],
                    "returns": {"type": "str", "description": "Analysis conclusion from the LLM."}
                }
                # Could add other skills here if EchoAgent develops more capabilities
            ]
        }

    async def execute(self, wsm_data: Any, **kwargs: Any) -> Any:
        """
        Main execution entry point for EchoAgent, fulfilling AgentBase contract.
        This will call the synchronous analyze_simulation_data method.
        """
        agent_name = self.config.get('name', type(self).__name__)
        self.logger.info(f"EchoAgent '{agent_name}' executing with MCP context and WSM data.")

        if self.context: # self.context is from AgentBase
            self.logger.info(f"Current context for '{agent_name}': {self.context}")

        # analyze_simulation_data is synchronous.
        # If the underlying llm_plugin call were async, this would need to be `await self.analyze_simulation_data(...)`
        # and analyze_simulation_data itself would need to be async.
        # For now, as per prompt, analyze_simulation_data is sync and LLMPlugin is sync.
        # If this needs to be non-blocking for a larger async system, consider asyncio.to_thread (Python 3.9+)
        # or loop.run_in_executor for older Python versions.

        try:
            # Direct call as analyze_simulation_data is synchronous
            result = self.analyze_simulation_data(wsm_data)
            self.logger.info(f"EchoAgent '{agent_name}' execution completed.")
            return result
        except Exception as e:
            self.logger.error(f"Exception during EchoAgent '{agent_name}' execution: {e}", exc_info=True)
            # Depending on error handling strategy, might raise or return an error structure
            return {"error": str(e), "details": "Failed during EchoAgent.execute"}

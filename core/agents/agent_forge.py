# core/agents/agent_forge.py
import logging
import os
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import importlib
import yaml

from core.agents.agent_base import AgentBase
from core.utils.config_utils import load_config, save_config


class AgentForge(AgentBase):
    """
    The AgentForge is responsible for the dynamic creation of new agents.
    It uses templates and configuration to generate agent code and add them
    to the system at runtime. This version incorporates advanced features
    like skill schema generation and A2A wiring.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.template_dir = Path(self.config.get("template_dir", "core/agents/templates"))
        self.agent_config_path = Path("config/agents.yaml")
        self.workflows_config_path = Path("config/workflows.yaml")
        self.agent_classes = self.load_agent_classes()

        logging.info(f"AgentForge initialized with template dir: {self.template_dir}")

    def load_agent_classes(self) -> Dict[str, str]:
        """
        Loads the available agent classes for extension.
        This could be read from a config file or dynamically discovered.
        For simplicity, it's hardcoded here.
        """
        return {
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

    async def execute(self, action: str, **kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Executes an agent creation action.
        """

        if action == "create_agent":
            return await self.create_agent(
                kwargs.get("agent_name"),
                kwargs.get("agent_type"),
                kwargs.get("agent_description"),
                kwargs.get("agent_skills"),
                kwargs.get("agent_dependencies"),
                kwargs.get("agent_a2a_peers"),
            )
        elif action == "list_templates":
            return self.list_templates()
        elif action == "get_template":
            return self.get_template(kwargs.get("template_name"))
        else:
            logging.warning(f"AgentForge: Unknown action: {action}")
            return None

    def list_templates(self) -> List[str]:
        """
        Lists available agent templates.
        """
        return [f.stem for f in self.template_dir.glob("*.py")]

    def get_template(self, template_name: str) -> Optional[str]:
        """
        Retrieves the content of a specific agent template.
        """
        template_path = self.template_dir / f"{template_name}.py"
        if template_path.exists():
            return template_path.read_text()
        else:
            logging.warning(f"AgentForge: Template not found: {template_name}")
            return None

    async def create_agent(
        self,
        agent_name: str,
        agent_type: str,
        agent_description: str,
        agent_skills: List[Dict[str, Any]],
        agent_dependencies: List[str],
        agent_a2a_peers: List[str],
    ) -> Optional[str]:
        """
        Creates a new agent.
        """

        try:
            # 1. Select Template
            template_name = self.config.get("default_template", "basic_agent")  # Default template
            template = self.get_template(template_name)
            if not template:
                raise ValueError(f"Agent template '{template_name}' not found.")

            # 2. Customize Template
            customized_code = self.customize_template(
                template, agent_name, agent_description, agent_skills, agent_a2a_peers
            )

            # 3. Validate & Optimize (Placeholder - CodeAlchemist agent?)
            validated_code = await self.validate_and_optimize_code(customized_code)
            if not validated_code:
                raise ValueError("Generated code is invalid.")

            # 4. Save Agent Code
            agent_filename = f"{agent_name.lower()}.py"
            agent_path = Path("core/agents") / agent_filename
            self.save_agent_code(validated_code, agent_path)

            # 5. Update Configurations
            self.update_agent_config(agent_name, agent_type, agent_description, agent_dependencies, agent_a2a_peers)
            self.update_workflows_config(agent_name, agent_dependencies)

            # 6. Agent Initialization (Orchestrator handles this)
            # Orchestrator will reload agents or dynamically import the new agent

            return f"Agent '{agent_name}' created successfully in '{agent_path}'."

        except Exception as e:
            logging.error(f"AgentForge: Error creating agent '{agent_name}': {e}")
            return None

    def customize_template(
        self,
        template: str,
        agent_name: str,
        agent_description: str,
        agent_skills: List[Dict[str, Any]],
        agent_a2a_peers: List[str],
    ) -> str:
        """
        Customizes the agent template with agent-specific details.
        This includes generating the skill schema and A2A wiring.
        """

        # Replace placeholders in the template
        code = (
            template.replace("CLASS_NAME", agent_name)
            .replace("AGENT_DESCRIPTION", agent_description)
        )

        # Generate skill schema code
        skill_schema_code = self.generate_skill_schema_code(agent_skills)
        code = code.replace("SKILL_SCHEMA", skill_schema_code)

        # Generate A2A wiring code
        a2a_wiring_code = self.generate_a2a_wiring_code(agent_a2a_peers)
        code = code.replace("A2A_WIRING", a2a_wiring_code)

        return code

    def generate_skill_schema_code(self, agent_skills: List[Dict[str, Any]]) -> str:
        """
        Generates the code for the get_skill_schema() method based on the
        provided agent skills.
        """

        skills_code = ",\n            ".join([f"""
                {{
                    "name": "{skill['name']}",
                    "description": "{skill['description']}",
                    "inputs": {json.dumps(skill.get('inputs', []))},
                    "outputs": {json.dumps(skill.get('outputs', []))}
                }}"""
            for skill in agent_skills])

        return f"""
        def get_skill_schema(self) -> Dict[str, Any]:
            return {{
                "name": type(self).__name__,
                "description": self.config.get("description", "No description provided"),
                "skills": [
                    {skills_code}
                ]
            }}
        """

    def generate_a2a_wiring_code(self, agent_a2a_peers: List[str]) -> str:
        """
        Generates the code to add peer agents for A2A communication.
        This assumes the AgentOrchestrator will call the add_peer_agent method.
        """

        wiring_code = "\n        ".join(
            [f"self.add_peer_agent('{peer_agent}');" for peer_agent in agent_a2a_peers]
        )
        return f"""
        def wire_a2a_peers(self):
            {wiring_code}
        """

    async def validate_and_optimize_code(self, code: str) -> Optional[str]:
        """
        Placeholder for code validation and optimization using a CodeAlchemist agent.
        In a real system, this would involve:
            - Syntax checking
            - Security scanning
            - Performance analysis
            - Code formatting
        """

        # For now, just return the code as is (for demonstration purposes)
        return code
        # Example of calling a CodeAlchemist agent (if available):
        # code_alchemist = self.orchestrator.get_agent("CodeAlchemist")
        # if code_alchemist:
        #    return await code_alchemist.execute(action="validate_and_optimize", code=code)
        # else:
        #    logging.warning("CodeAlchemist agent not available. Skipping code validation and optimization.")
        #    return code

    def save_agent_code(self, code: str, agent_path: Path) -> None:
        """
        Saves the generated agent code to a file.
        """

        try:
            with open(agent_path, "w") as f:
                f.write(code)
            logging.info(f"Agent code saved to: {agent_path}")
        except Exception as e:
            logging.error(f"AgentForge: Error saving agent code to '{agent_path}': {e}")
            raise

    def update_agent_config(
        self,
        agent_name: str,
        agent_type: str,
        agent_description: str,
        agent_dependencies: List[str],
        agent_a2a_peers: List[str],
    ) -> None:
        """
        Updates the agent configuration file (config/agents.yaml) to include the new agent.
        """

        agent_config = load_config(self.agent_config_path)
        if agent_config is None:
            logging.error(f"AgentForge: Could not load agent config from {self.agent_config_path}")
            return

        agent_config[agent_name] = {
            "type": agent_type,
            "description": agent_description,
            "dependencies": agent_dependencies,
            "a2a_peers": agent_a2a_peers,
        }
        save_config(self.agent_config_path, agent_config)
        logging.info(f"Agent config updated with: {agent_name}")

    def update_workflows_config(self, agent_name: str, agent_dependencies: List[str]) -> None:
        """
        Updates the workflows configuration file (config/workflows.yaml) to
        incorporate the new agent into relevant workflows (if any).
        This is a simplified example; a more robust implementation would
        involve more intelligent workflow modification.
        """

        workflows_config = load_config(self.workflows_config_path)
        if workflows_config is None:
            logging.error(f"AgentForge: Could not load workflows config from {self.workflows_config_path}")
            return

        for workflow in workflows_config.values():
            if agent_dependencies:
                # Naive: Add the new agent to any workflow that uses its dependencies
                if any(dep in workflow["agents"] for dep in agent_dependencies):
                    workflow["agents"].append(agent_name)
                    workflow["dependencies"][agent_name] = agent_dependencies
                    logging.info(f"Workflows config updated to include {agent_name}")
                    break  # Only update the first matching workflow (for simplicity)

        save_config(self.workflows_config_path, workflows_config)

    async def run_semantic_kernel_skill(self, skill_name: str, input_vars: Dict[str, str]) -> str:
        """
        Placeholder for running a Semantic Kernel skill.
        This assumes the AgentForge might use SK for code generation or validation.
        """

        # For now, just return a dummy result
        return f"Semantic Kernel skill '{skill_name}' executed (dummy result)."

        # Example of calling a Semantic Kernel skill (if available):
        # if hasattr(self, 'kernel') and self.kernel:
        #    return await self.kernel.run(skill_name, input_vars=input_vars)
        # else:
        #    logging.warning("Semantic Kernel not available in AgentForge.")
        #    return f"Semantic Kernel not available. Cannot execute skill '{skill_name}'."

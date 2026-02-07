from typing import Dict, Any, List
import logging
import json
import os
import importlib
import inspect
from core.agents.agent_base import AgentBase
from core.agents.mixins.audit_mixin import AuditMixin

class SkillHarvesterAgent(AgentBase, AuditMixin):
    """
    Crawls the agent swarm to extract 'get_skill_schema' definitions
    and compiles a structured registry JSON for the Agent Gallery and MCP.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        AuditMixin.__init__(self)
        self.registry_path = "docs/agent_skills_registry.json"

    async def execute(self, **kwargs) -> Dict[str, Any]:
        logging.info("SkillHarvesterAgent starting harvest...")

        agents = self._find_agent_classes()
        registry = []

        for agent_cls in agents:
            try:
                # Instantiate lightly to get schema
                # We assume a default config is enough
                agent = agent_cls(config={"name": "harvester_temp"})
                schema = agent.get_skill_schema()
                if schema:
                    registry.append(schema)
            except Exception as e:
                logging.warning(f"Could not harvest skills from {agent_cls.__name__}: {e}")

        # Save Registry
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        self.log_decision(
            activity_type="SkillHarvest",
            details={"agents_scanned": len(agents)},
            outcome={"registry_size": len(registry)}
        )

        return {"status": "SUCCESS", "registry_path": self.registry_path}

    def _find_agent_classes(self) -> List[type]:
        """
        Dynamically finds all AgentBase subclasses in core/agents.
        """
        found_agents = []
        root_dir = "core/agents"

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith("_agent.py"):
                    module_path = os.path.join(root, file).replace("/", ".").replace(".py", "")
                    try:
                        module = importlib.import_module(module_path)
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, AgentBase) and obj is not AgentBase:
                                found_agents.append(obj)
                    except Exception as e:
                        pass

        return list(set(found_agents))

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "SkillHarvesterAgent",
            "description": "Compiles the central skill registry.",
            "skills": [
                {
                    "name": "harvest_skills",
                    "description": "Updates agent_skills_registry.json",
                    "parameters": {}
                }
            ]
        }

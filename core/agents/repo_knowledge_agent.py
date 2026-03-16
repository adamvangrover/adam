import logging
import os
from typing import Any, Dict

from core.agents.agent_base import AgentBase


class RepoKnowledgeAgent(AgentBase):
    """
    Scans the repository structure and documentation to maintain
    'System Health' insights in the Swarm Memory.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        # core/agents/repo_knowledge_agent.py -> core/agents -> core -> root
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.logger = logging.getLogger(__name__)

    async def execute(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a repository scan and publishes insights.
        """
        self.logger.info("RepoKnowledgeAgent: Scanning repository...")

        insights = {}

        # 1. Agent Census
        agent_count = 0
        agents_dir = os.path.join(self.root_dir, 'core', 'agents')
        if os.path.exists(agents_dir):
            for root, dirs, files in os.walk(agents_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        agent_count += 1

        insight_agents = f"Detected {agent_count} agent modules active in the core architecture."
        self.publish_insight("System Architecture", insight_agents, 0.95)
        insights["agent_count"] = agent_count

        # 2. Module Census (Core)
        core_files = 0
        core_dir = os.path.join(self.root_dir, 'core')
        if os.path.exists(core_dir):
            for root, dirs, files in os.walk(core_dir):
                for file in files:
                    if file.endswith('.py'):
                        core_files += 1

        insight_modules = f"Core system footprint: {core_files} python modules."
        self.publish_insight("System Architecture", insight_modules, 0.9)
        insights["core_files"] = core_files

        # 3. Compliance Check (AGENTS.md)
        agents_md = os.path.join(self.root_dir, 'AGENTS.md')
        if os.path.exists(agents_md):
            self.publish_insight("Compliance", "AGENTS.md protocol file found. Governance active.", 1.0)
            insights["compliance"] = "Active"
        else:
            self.publish_insight("Compliance", "CRITICAL: AGENTS.md missing. Governance protocols inactive.", 0.8)
            insights["compliance"] = "Missing"

        # 4. Readme Check
        readme_md = os.path.join(self.root_dir, 'README.md')
        if os.path.exists(readme_md):
             self.publish_insight("Documentation", "System documentation (README.md) verified.", 0.9)

        return insights

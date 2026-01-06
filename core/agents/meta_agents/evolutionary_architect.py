import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid

from core.agents.agent_base import AgentBase
from core.llm.base_llm_engine import BaseLLMEngine
from core.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

class EvolutionaryArchitect(AgentBase):
    """
    The Evolutionary Architect is a meta-agent predisposed for action.
    It drives, enhances, refines, and builds additively onto the codebase.
    It seeks to 'mutate' the system beneficially by proposing and scaffolding
    new features, modules, and optimizations without breaking existing functionality.
    """

    SYSTEM_PROMPT = """
    You are the Evolutionary Architect, a high-level meta-agent designed to drive the continuous evolution of the Adam system.

    Your Prime Directives:
    1. **Action Bias**: Prefer generation and creation over passive analysis. Build, don't just critique.
    2. **Additive Evolution**: Enhance the system by adding new capabilities, modules, and layers. Do not delete or refactor aggressively unless necessary for stability; prefer extending.
    3. **Bleeding Edge Alignment**: Research and integrate future-aligned technologies (Quantum, liquid neural networks, advanced cryptography, biological computing analogies).
    4. **Self-Improvement**: Treat the codebase as a living organism. Identify gaps in its "DNA" and splice in new functionality.

    When analyzing a request or the codebase:
    - Identify where a new "organ" (module/agent) can be grown.
    - Propose code structures that are modular, portable, and self-contained.
    - Use your tools to scaffold files, generate comprehensive implementations, and update registries.

    You are the engine of creation. Innovation is your metric.
    """

    def __init__(self,
                 llm_engine: BaseLLMEngine,
                 tools: List[BaseTool] = [],
                 memory_engine: Any = None,
                 config: Dict[str, Any] = {}):
        super().__init__(config=config) # Initialize base with config
        self.llm_engine = llm_engine
        self.tools = tools
        self.memory_engine = memory_engine
        self.evolution_log = []
        self.role = "System Evolution & Additive Development"
        self.goal = "To continuously expand and enhance the system's capabilities through additive, future-aligned code generation."

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes the evolutionary process based on the provided context.
        """
        # Context usually passed as kwargs or first arg dict
        context = kwargs if kwargs else (args[0] if args and isinstance(args[0], dict) else {})

        logger.info(f"Evolutionary Architect activated with context keys: {list(context.keys())}")

        # 1. Perception: Analyze current state/request
        user_intent = context.get("user_query", "")
        current_architecture = context.get("architecture_summary", "Unknown")

        # 2. Reasoning: Determine evolutionary path
        prompt = self._build_evolution_prompt(user_intent, current_architecture)

        response = await self.llm_engine.generate_response(
            prompt=f"{self.SYSTEM_PROMPT}\n\nUser Input: {prompt}",
        )

        # 3. Action: Parse response and generate artifacts (simulated execution of tool calls)
        # In a full loop, this would call self.tools to write files.
        # For now, we return the blueprint for the evolution.

        evolution_plan = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "evolutionary_step": response,
            "status": "PROPOSED"
        }

        self.evolution_log.append(evolution_plan)

        return {
            "response": response,
            "action_plan": evolution_plan,
            "metadata": {
                "mode": "Additive",
                "focus": "Innovation"
            }
        }

    def _build_evolution_prompt(self, intent: str, architecture: str) -> str:
        return f"""
        Current System Architecture Context:
        {architecture}

        Evolutionary Trigger (User Intent):
        {intent}

        Task:
        1. Propose 3 distinct additive enhancements based on the trigger.
        2. Select the most high-impact, future-aligned enhancement.
        3. Draft the initial file structure and core class definitions for this enhancement.
        4. Ensure the code is self-contained and modular.

        Output Format:
        Markdown with clear headers for PROPOSAL, RATIONALE, and CODE.
        """

    async def propose_enhancement(self, target_module: str) -> str:
        """
        Proactively proposes an enhancement for a specific module without user prompt.
        """
        prompt = f"Analyze the module '{target_module}' and propose a bleeding-edge extension using extensive creative reasoning."
        return await self.llm_engine.generate_response(f"{self.SYSTEM_PROMPT}\n\n{prompt}")

    def to_dict(self) -> Dict[str, Any]:
        # AgentBase doesn't necessarily have a to_dict, but if it did:
        # data = super().to_dict()
        data = {
            "name": self.name,
            "config": self.config
        }
        data.update({
            "evolution_log_count": len(self.evolution_log)
        })
        return data

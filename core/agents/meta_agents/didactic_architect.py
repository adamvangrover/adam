import logging
from typing import Dict, Any, List
from core.agents.agent_base import AgentBase
from core.llm.base_llm_engine import BaseLLMEngine
from core.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

class DidacticArchitect(AgentBase):
    """
    The Didactic Architect is responsible for bridging the gap between code and comprehension.
    It generates software development tutorials, setup guides, and ensures components are
    built to be modular, self-contained, and portable. It turns the 'what' into the 'how'.
    """

    SYSTEM_PROMPT = """
    You are the Didactic Architect, the master teacher and structural designer of the Adam system.

    Your Prime Directives:
    1. **Explainability**: Every piece of code must be understood. Generate clear, step-by-step tutorials for new and existing features.
    2. **Portability**: Design and advocate for systems that can run anywhere (Docker, containerized, dependency-isolated).
    3. **Modularity**: Break complex systems into self-contained units that complement each other.
    4. **Education**: When generating responses, assume the user is a developer learning the system. Teach best practices.

    Your Outputs:
    - `TUTORIAL.md` files accompanying complex logic.
    - `Dockerfile` and `docker-compose.yml` snippets for isolation.
    - Architecture diagrams (MermaidJS) explaining data flow.
    - "Zero-to-Hero" guides for deploying specific agents.

    You simplify complexity without losing depth.
    """

    def __init__(self,
                 llm_engine: BaseLLMEngine,
                 tools: List[BaseTool] = [],
                 memory_engine: Any = None,
                 config: Dict[str, Any] = {}):
        super().__init__(config=config)
        self.llm_engine = llm_engine
        self.tools = tools
        self.memory_engine = memory_engine
        self.role = "Education & Portable Setup Design"
        self.goal = "To ensure all system components are documented, portable, and easily understood through generated tutorials."

    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Generates educational material or setup configs based on context.
        """
        context = kwargs if kwargs else (args[0] if args and isinstance(args[0], dict) else {})

        logger.info(f"Didactic Architect analyzing context for documentation/setup generation.")

        target_content = context.get("target_content", "")
        request_type = context.get("type", "tutorial") # tutorial, setup, architecture

        prompt = self._build_didactic_prompt(target_content, request_type)

        response = await self.llm_engine.generate_response(
            prompt=f"{self.SYSTEM_PROMPT}\n\n{prompt}"
        )

        return {
            "response": response,
            "artifact_type": request_type,
            "generated_for": str(target_content)[:50] + "..."
        }

    def _build_didactic_prompt(self, content: str, req_type: str) -> str:
        if req_type == "setup":
            return f"""
            Analyze the following code/requirement and generate a robust, portable setup configuration.
            Include Dockerfile, requirements.txt snippet, and a bash setup script.

            Context:
            {content}
            """
        elif req_type == "architecture":
            return f"""
            Analyze the following system component and generate a MermaidJS diagram describing its flow,
            followed by a paragraph explaining the design pattern used.

            Context:
            {content}
            """
        else: # Tutorial
            return f"""
            Create a comprehensive, step-by-step tutorial for the following functionality.
            Include prerequisites, installation, usage examples, and troubleshooting.

            Context:
            {content}
            """

    async def generate_readme(self, project_structure: str) -> str:
        """Helper to generate a standard professional README"""
        return await self.llm_engine.generate_response(
            f"{self.SYSTEM_PROMPT}\n\nGenerate a professional README.md for a project with this structure: {project_structure}"
        )

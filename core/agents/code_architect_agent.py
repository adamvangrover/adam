import logging
import os
import re
from typing import Any, Dict

from core.agents.pydantic_agent_base import PydanticAgentBase
from core.schemas.agent_schema import AgentInput, AgentOutput


class CodeArchitectAgent(PydanticAgentBase):
    """
    Agent for dynamically generating UI visualizations using Chart.js
    and Vanilla CSS (Glassmorphism), saving executable outputs safely.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.persona = "Code Architect"
        self.description = "Generates UI visualizations and standalone data pipelines."
        self.save_dir = "scripts"

    async def execute_pydantic(self, input_data: AgentInput) -> AgentOutput:
        input_data.query.strip()
        context = input_data.context

        # Placeholder logic: usually we would query an LLM to generate the code.
        # Here we mock the behavior for demonstration and test the constraints.

        filename = context.get("filename", "visualization.html")
        code_content = context.get("code", "<html><body>Generated UI</body></html>")

        # Sanitize filename to prevent path traversal
        if not re.match(r'^[\w\-. ]+$', filename):
            return AgentOutput(
                answer="Failed: Invalid filename. Path traversal suspected.",
                confidence=0.0,
                metadata={"error": "Path traversal check failed"}
            )

        # Protect Index_archive.html
        safe_override = context.get("safe_override", False)
        if filename == "Index_archive.html" and not safe_override:
            return AgentOutput(
                answer="Failed: Explicit routing safety check required to overwrite Index_archive.html.",
                confidence=0.0,
                metadata={"error": "Index_archive.html overwrite protection triggered"}
            )

        filepath = os.path.join(self.save_dir, filename)

        try:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(code_content)

            return AgentOutput(
                answer=f"Successfully generated code and saved to {filepath}",
                sources=[filepath],
                confidence=1.0,
                metadata={"filepath": filepath, "content_length": len(code_content)}
            )
        except Exception as e:
            logging.exception(f"CodeArchitect error saving to {filepath}")
            return AgentOutput(
                answer=f"Execution failed: {e}",
                confidence=0.0,
                metadata={"error": str(e)}
            )

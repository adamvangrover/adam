
import json
import logging
import os
import re
from typing import Any, Dict, Optional

from core.agents.agent_base import AgentBase
from core.schemas.credit_conformance import CreditConformanceReport
from core.llm_plugin import LLMPlugin
from pydantic import ValidationError

try:
    from semantic_kernel import Kernel
except ImportError:
    Kernel = Any

class CreditConformanceAgent(AgentBase):
    """
    Tier-2 Generative AI Agent for Credit Risk Conformance.
    Implements a multi-layered architecture for regulatory and policy conformance.
    """

    def __init__(self, config: Dict[str, Any], constitution: Optional[Dict[str, Any]] = None, kernel: Optional[Kernel] = None):
        super().__init__(config, constitution, kernel)
        self.llm_plugin = LLMPlugin(config=config.get("model_config", {}))
        self.prompt_path = "prompt_library/AOPL-v1.0/professional_outcomes/LIB-PRO-008_credit_conformance_tier2.md"
        self.max_retries = 3

    def _load_prompt(self) -> str:
        """Loads the master prompt from file."""
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt file not found at {self.prompt_path}")
        with open(self.prompt_path, "r") as f:
            return f.read()

    async def execute(self, document_text: str, policy_text: str, **kwargs) -> Dict[str, Any]:
        """
        Executes the credit conformance review.

        Args:
            document_text: The full text of the credit document.
            policy_text: The text of the policies to check against.

        Returns:
            A dictionary representation of the CreditConformanceReport.
        """
        logging.info("Starting Credit Conformance Review...")

        master_prompt_template = self._load_prompt()

        # Simple string replacement
        prompt = master_prompt_template.replace("{{ document_under_review }}", document_text)
        prompt = prompt.replace("{{ policy_standards }}", policy_text)

        response_text = ""
        for attempt in range(self.max_retries):
            try:
                # Call LLM synchronously as LLMPlugin is synchronous
                response_text = self.llm_plugin.generate_text(prompt)

                # Extract JSON from response (handle code blocks)
                json_str = self._extract_json(response_text)

                # Parse and Validate with Pydantic
                report = CreditConformanceReport.model_validate_json(json_str)

                logging.info(f"Conformance Review successful. Status: {report.reportMetadata.overallConformanceStatus}")
                return report.model_dump(by_alias=True)

            except (ValidationError, json.JSONDecodeError) as e:
                logging.warning(f"Attempt {attempt + 1} failed validation: {e}")
                if attempt == self.max_retries - 1:
                    logging.error("Max retries reached. Returning raw response and error.")
                    return {
                        "error": "Failed to generate valid report",
                        "details": str(e),
                        "raw_response": response_text
                    }
            except Exception as e:
                logging.error(f"Unexpected error during execution: {e}")
                raise e

        return {"error": "Unknown failure"}

    def _extract_json(self, text: str) -> str:
        """Extracts JSON substring from the text, handling markdown code blocks."""
        # Check for markdown code blocks
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1)

        # Fallback: find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start:end+1]

        return text

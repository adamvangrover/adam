from __future__ import annotations

import re

from core.prompting.base_prompt_plugin import BasePromptPlugin
from core.schemas.financial_truth import FinancialTruthInput, FinancialTruthOutput


class FinancialTruthPlugin(BasePromptPlugin[FinancialTruthOutput]):
    """
    A prompt plugin that implements the FinanceBench/TAO "System 2" reasoning framework
    for high-precision financial auditing.
    """

    def get_input_schema(self):
        return FinancialTruthInput

    def get_output_schema(self):
        return FinancialTruthOutput

    def render_messages(self, data: dict) -> list[dict[str, str]]:
        """
        Renders the prompt template with the provided data (context and question).
        """
        # Validate input data
        validated_data = self.get_input_schema()(**data)

        # Render the template using the BasePromptPlugin's Jinja2 engine
        # The placeholders in the markdown are {{CONTEXT}} and {{QUESTION}}
        user_content = self.user_engine.render(
            CONTEXT=validated_data.context,
            QUESTION=validated_data.question
        )

        return [{"role": "user", "content": user_content}]

    def parse_response(self, response: str) -> FinancialTruthOutput:
        """
        Parses the raw LLM response string into the structured Information Triplet.
        The expected format is:
        <thinking>...</thinking>
        **Answer:** ...
        **Evidence:** ...
        **Logic:** ...
        """
        # Remove the <thinking> block for cleaner parsing, but keep it if logging is needed later.
        # For the Output schema, we just need Answer, Evidence, Logic.

        # Regex patterns to extract the triplet
        # We use re.DOTALL to handle multiline content
        answer_pattern = r"\*\*Answer:\*\*\s*(.+?)(?=\n\*\*Evidence:|$)"
        evidence_pattern = r"\*\*Evidence:\*\*\s*(.+?)(?=\n\*\*Logic:|$)"
        logic_pattern = r"\*\*Logic:\*\*\s*(.+?)(?=$)"

        answer_match = re.search(answer_pattern, response, re.DOTALL)
        evidence_match = re.search(evidence_pattern, response, re.DOTALL)
        logic_match = re.search(logic_pattern, response, re.DOTALL)

        answer = answer_match.group(1).strip() if answer_match else "Parse Error: Answer not found."
        evidence = evidence_match.group(1).strip() if evidence_match else "Parse Error: Evidence not found."
        logic = logic_match.group(1).strip() if logic_match else "Parse Error: Logic not found."

        # If strict parsing fails, we might want to return the raw response as 'Logic' or 'Answer'
        # but for now, we follow the schema.

        return FinancialTruthOutput(
            answer=answer,
            evidence=evidence,
            logic=logic
        )

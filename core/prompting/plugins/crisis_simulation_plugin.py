from __future__ import annotations
import json
from core.prompting.base_prompt_plugin import BasePromptPlugin
from core.schemas.crisis_simulation import CrisisSimulationInput, CrisisSimulationOutput


class CrisisSimulationPlugin(BasePromptPlugin[CrisisSimulationOutput]):
    """
    A prompt plugin for running enterprise-grade crisis simulations based on a
    structured risk portfolio and a user-defined scenario.
    """

    def get_input_schema(self):
        return CrisisSimulationInput

    def get_output_schema(self):
        return CrisisSimulationOutput

    def render_messages(self, data: dict) -> list[dict[str, str]]:
        """
        Renders the prompt template with the provided data, structuring it
        for a chat-based LLM.
        """
        # Validate input data against the schema
        validated_data = self.get_input_schema()(**data)

        # Convert the Pydantic models to JSON strings for injection
        risk_portfolio_json = json.dumps([risk.model_dump(by_alias=True)
                                         for risk in validated_data.risk_portfolio], indent=4)

        # Render the user template (which contains the full XML structure)
        user_content = self.user_engine.render(
            RISK_PORTFOLIO_JSON=risk_portfolio_json,
            CURRENT_DATE=validated_data.current_date,
            USER_SCENARIO_INPUT=validated_data.user_scenario
        )

        # The system message is integrated into the user message via XML tags,
        # so we only need a single user message here.
        return [{"role": "user", "content": user_content}]

    def parse_response(self, response: str) -> CrisisSimulationOutput:
        """
        Parses the raw LLM response string into the structured output schema.
        """
        # The base implementation attempts to parse JSON, which is what we need.
        return super().parse_response(response)

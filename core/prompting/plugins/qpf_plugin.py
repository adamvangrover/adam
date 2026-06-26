from __future__ import annotations
import re
from core.prompting.base_prompt_plugin import BasePromptPlugin
from core.schemas.qpf_schema import QPFInput, QPFOutput

class QPFPlugin(BasePromptPlugin[QPFOutput]):
    """
    Quant Prompt Framework (QPF) plugin for high-fidelity quantitative analysis prompts.
    """
    def get_input_schema(self) -> type[QPFInput]:
        return QPFInput

    def get_output_schema(self) -> type[QPFOutput]:
        return QPFOutput

    def render_messages(self, data: dict) -> list[dict[str, str]]:
        validated_data = self.get_input_schema()(**data)

        user_content = self.user_engine.render(
            objective=validated_data.objective,
            universe=validated_data.universe,
            data_frequency=validated_data.data_frequency,
            methodology=validated_data.methodology,
            deliverable=validated_data.deliverable,
            risk_metrics=validated_data.risk_metrics,
            constraints=validated_data.constraints
        )
        return [{"role": "user", "content": user_content}]

    def parse_response(self, response: str) -> QPFOutput:
        return super().parse_response(response)

from __future__ import annotations
from typing import List, Dict
from pydantic import BaseModel, Field
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata

class ExampleInput(BaseModel):
    user_name: str = Field(..., description="Name of the user")
    task_list: List[str] = Field(..., description="List of tasks to perform")

class ExampleOutput(BaseModel):
    summary: str = Field(..., description="Summary of the plan")
    estimated_time: int = Field(..., description="Estimated time in minutes")

class ExamplePlugin(BasePromptPlugin[ExampleOutput]):
    """
    An example plugin demonstrating the 'Prompt-as-Code' pattern.
    """

    def get_input_schema(self):
        return ExampleInput

    def get_output_schema(self):
        return ExampleOutput

    # Optional: Override render if you need custom logic beyond Jinja2
    # def render(self, inputs: Dict[str, Any]) -> str:
    #     ...

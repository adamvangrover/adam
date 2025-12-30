from typing import List, Dict, Any, Type
from pydantic import BaseModel, Field
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata

# --- Schemas ---

class SelfDiscoverInput(BaseModel):
    task_description: str
    context: str

class SelfDiscoverStructure(BaseModel):
    reasoning_modules: List[str] = Field(..., description="List of reasoning modules selected (e.g., 'Critical Thinking', 'Step-by-step', 'Fact Check').")
    plan: str = Field(..., description="The discovered reasoning structure/plan.")

class CoVeInput(BaseModel):
    query: str
    initial_response: str

class CoVeVerification(BaseModel):
    verification_questions: List[str] = Field(..., description="Questions to verify facts in the initial response.")
    verified_facts: Dict[str, str] = Field(..., description="Mapping of question to verified answer.")
    final_response: str = Field(..., description="The corrected response based on verification.")

# --- Plugins ---

class SelfDiscoverPrompt(BasePromptPlugin[SelfDiscoverStructure]):
    """
    Implements the 'Self-Discover' prompting framework (DeepMind).
    Phase 1: SELECT reasoning modules.
    Phase 2: ADAPT them to the task.
    Phase 3: IMPLEMENT the reasoning structure.

    This plugin handles the generation of the Structure (Phase 1-2).
    """

    def __init__(self):
        super().__init__(
            metadata=PromptMetadata(
                prompt_id="self_discover_v1",
                author="google_deepmind_paper_impl",
                model_config={"temperature": 0.2}
            ),
            user_template=(
                "Task: {{ task_description }}\n"
                "Context: {{ context }}\n\n"
                "You are an expert problem solver using the Self-Discover framework.\n"
                "1. Select relevant reasoning modules (e.g., Decomposition, Critical Thinking, Creative Thinking).\n"
                "2. Adapt these modules to the specific task.\n"
                "3. Create a structured reasoning plan.\n\n"
                "Output the result in JSON format with 'reasoning_modules' and 'plan'."
            )
        )

    def get_input_schema(self) -> Type[BaseModel]:
        return SelfDiscoverInput

    def get_output_schema(self) -> Type[SelfDiscoverStructure]:
        return SelfDiscoverStructure


class ChainOfVerificationPrompt(BasePromptPlugin[CoVeVerification]):
    """
    Implements Chain-of-Verification (CoVe) logic.
    1. Draft initial response (input).
    2. Plan verification questions.
    3. Execute verification (simulated here as generating answers).
    4. Generate final verified response.
    """

    def __init__(self):
        super().__init__(
            metadata=PromptMetadata(
                prompt_id="cove_v1",
                author="google_research_impl",
                model_config={"temperature": 0.1}
            ),
            user_template=(
                "Original Query: {{ query }}\n"
                "Initial Response: {{ initial_response }}\n\n"
                "Perform a Chain-of-Verification:\n"
                "1. Identify independent facts in the initial response that need verification.\n"
                "2. Formulate verification questions for these facts.\n"
                "3. Answer these questions (simulate the verification) to check for hallucinations.\n"
                "4. Produce a final, corrected response.\n\n"
                "Output JSON with 'verification_questions', 'verified_facts', and 'final_response'."
            )
        )

    def get_input_schema(self) -> Type[BaseModel]:
        return CoVeInput

    def get_output_schema(self) -> Type[CoVeVerification]:
        return CoVeVerification

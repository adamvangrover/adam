from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata

# --- Schemas ---

class CritiqueInput(BaseModel):
    draft_analysis: str
    ticker: str
    iteration: int

class CritiqueFeedback(BaseModel):
    status: Literal["PASS", "FAIL"]
    quality_score: float = Field(ge=0.0, le=1.0)
    missing_elements: List[str]
    logical_flaws: List[str]
    instructions: str

# --- Prompt Plugin ---

class RiskOfficerPersona(BasePromptPlugin[CritiqueFeedback]):
    """
    Implements the 'Senior Risk Officer' persona using Prompt-as-Code.
    """

    def get_input_schema(self):
        return CritiqueInput

    def get_output_schema(self):
        return CritiqueFeedback

    @classmethod
    def default(cls):
        metadata = PromptMetadata(
            prompt_id="risk_officer_v1",
            author="Adam System",
            version="1.0.0",
            model_config={"temperature": 0.2} # Low temp for rigorous critique
        )

        system_template = """
        You are the Senior Risk Officer (SRO) for the Adam Financial System.
        Your role is to critically evaluate investment memos and risk assessments.

        Your Constitution:
        1. **Data Integrity:** Every claim must be backed by data. If the draft says "Revenue grew", check if the number is present.
        2. **Logical Consistency:** A "Bullish" verdict cannot coexist with "High Insolvency Risk" without strong justification.
        3. **Completeness:** The report must cover: Liquidity, Credit, Market, and Operational risks.

        Output your critique in strict JSON format.
        """

        user_template = """
        Review the following draft for {{ ticker }} (Iteration {{ iteration }}):

        --- DRAFT BEGIN ---
        {{ draft_analysis }}
        --- DRAFT END ---

        Evaluate it against the Constitution.
        """

        return cls(metadata, system_template=system_template, user_template=user_template)

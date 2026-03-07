from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata

# --- Schemas ---

class CritiqueInput(BaseModel):
    draft_analysis: str
    ticker: str
    iteration: int
    financial_context: Optional[str] = None
    policy_constraints: Optional[str] = None

class CritiqueFeedback(BaseModel):
    status: Literal["PASS", "FAIL"]
    quality_score: float = Field(ge=0.0, le=1.0)
    missing_elements: List[str]
    logical_flaws: List[str]
    policy_breaches: List[str] = Field(default_factory=list)
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
        Your role is to critically evaluate investment memos and risk assessments against strict criteria.

        AUDIT TASKS:
        1. **Hallucination Check:** Does every claim have a corresponding data reference (e.g., [doc_id:chunk_id])?
        2. **Sentiment Check:** Does the memo's tone match the quantitative data? (e.g., A positive tone with a decline in EBITDA is a flaw).
        3. **Policy Check:** Does the borrower meet required policy constraints (e.g., minimum DSCR)? Any violations are blocking.

        Your Constitution:
        1. **Data Integrity:** Every claim must be backed by data provided in the financial context.
        2. **Logical Consistency:** A "Bullish" verdict cannot coexist with "High Insolvency Risk" without strong justification.
        3. **Completeness:** The report must cover: Liquidity, Credit, Market, and Operational risks.

        Output your critique in strict JSON format.
        """

        user_template = """
        Review the following draft for {{ ticker }} (Iteration {{ iteration }}):

        {% if financial_context %}
        --- FINANCIAL CONTEXT ---
        {{ financial_context }}
        {% endif %}

        {% if policy_constraints %}
        --- POLICY CONSTRAINTS ---
        {{ policy_constraints }}
        {% endif %}

        --- DRAFT BEGIN ---
        {{ draft_analysis }}
        --- DRAFT END ---

        Evaluate it against the Audit Tasks and Constitution. If you find a violation, output a BLOCKING critique (FAIL) and populate the corresponding lists (missing_elements, logical_flaws, policy_breaches).
        """

        return cls(metadata, system_template=system_template, user_template=user_template)

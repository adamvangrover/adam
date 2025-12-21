from pydantic import BaseModel, Field


class FinancialTruthInput(BaseModel):
    """
    Input schema for the Financial Truth TAO-CoT prompt.
    """
    context: str = Field(..., description="The financial context (e.g., retrieval chunks) to analyze.")
    question: str = Field(..., description="The specific financial question to answer.")


class FinancialTruthOutput(BaseModel):
    """
    Output schema for the Financial Truth TAO-CoT prompt.
    """
    answer: str = Field(..., description="The direct, concise answer to the question.")
    evidence: str = Field(..., description="Verbatim quote or table row from the text supporting the answer.")
    logic: str = Field(..., description="Brief explanation of the calculation or extraction method used.")

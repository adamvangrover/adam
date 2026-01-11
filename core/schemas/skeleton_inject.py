from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class SkeletonInput(BaseModel):
    context: str = Field(..., description="Raw RAG chunks containing Earnings Transcripts, 10-K MD&A, etc.")

class SkeletonOutput(BaseModel):
    skeleton_text: str = Field(..., description="Narrative template with {{PLACEHOLDERS}}.")

class SynthesisInput(BaseModel):
    draft_text: str = Field(..., description="The narrative text containing placeholders.")
    truth_data: Dict[str, Any] = Field(..., description="Dictionary of verified financial metrics.")

class SynthesisOutput(BaseModel):
    final_text: str = Field(..., description="The final, clean credit memo section.")

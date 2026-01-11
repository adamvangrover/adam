from typing import List, Optional
from pydantic import BaseModel, Field

class CritiqueInput(BaseModel):
    final_text: str = Field(..., description="The synthesized text to review.")

class CritiqueOutput(BaseModel):
    status: str = Field(..., description="APPROVED or REJECTED")
    score: int = Field(..., description="Quality score 0-100")
    feedback: str = Field(..., description="Qualitative feedback.")
    red_flags: List[str] = Field(default_factory=list, description="List of specific issues found.")

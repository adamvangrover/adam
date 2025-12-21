from pydantic import BaseModel
from typing import Dict, Any, Optional


class AnalysisRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = {}


class AnalysisResponse(BaseModel):
    status: str
    result: Any

from typing import Any, Dict, Optional

from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = {}

class AnalysisResponse(BaseModel):
    status: str
    result: Any

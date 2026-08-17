from typing import Dict, Any, List
from pydantic import BaseModel

class ContextManifest(BaseModel):
    """
    Defines the structural requirements for building a frozen context.
    """
    required_sources: List[str]
    required_market_data: bool
    required_portfolio_data: bool
    allowed_drift_ms: int = 5000 # Max time allowed between retrieval and execution
    metadata: Dict[str, Any]

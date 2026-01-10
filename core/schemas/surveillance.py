from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field

class DistressedIssuer(BaseModel):
    issuer_name: str
    sector: str
    primary_distress_signal: str
    debt_quantum: Optional[str] = None
    advisors: Optional[str] = None

class DistressedWatchlist(BaseModel):
    issuers: List[DistressedIssuer]

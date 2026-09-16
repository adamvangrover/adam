from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

class ProvenanceHeader(BaseModel):
    """W3C PROV-O compliant provenance header."""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    system_version: str = "30.1.0"
    observed_drift: bool = False
    execution_trace: List[str] = Field(default_factory=list)

class Obligor(BaseModel):
    """Obligor-level fundamental creditworthiness."""
    obligor_id: str
    name: str
    probability_of_default: float = Field(..., ge=0.0, le=1.0, description="Obligor-level PD")
    sector: str

class Facility(BaseModel):
    """Facility-level structure incorporating collateral and LGD."""
    facility_id: str
    obligor_id: str
    exposure_amount: float
    loss_given_default: float = Field(..., ge=0.0, le=1.0, description="Facility-level LGD")
    collateral_value: float

class RiskAssessmentOutput(BaseModel):
    provenance: ProvenanceHeader
    obligor_id: str
    facility_id: str
    expected_loss: float
    facility_rating: str

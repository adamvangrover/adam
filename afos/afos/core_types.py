from pydantic import BaseModel, Field, ConfigDict
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

class ModelArbitrationOutput(BaseModel):
    """Output schema for dual-model PD arbitration."""
    model_config = ConfigDict(strict=True, extra='forbid')

    bidirectional_disparity: float = Field(..., description="Absolute difference between challenger and baseline PDs.")
    arbitration_flag_triggered: bool = Field(..., description="Flag indicating if bidirectional divergence exceeded the threshold.")
    one_sided_downside_spread: float = Field(..., description="Max(0, PD_beta - PD_alpha).")
    capital_buffer_penalty: float = Field(..., description="Penalty applied based on the downside spread.")

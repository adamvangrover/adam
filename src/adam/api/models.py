from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OptimizerConfig(BaseModel):
    algorithm: str = Field(..., description="Optimizer name (adamw, lion, adam-mini)")
    learning_rate: float = Field(gt=0, description="Step size")
    betas: List[float] = Field(default=[0.9, 0.999], min_items=2, max_items=2)
    weight_decay: float = Field(default=0.0, ge=0)
    epsilon: float = Field(default=1e-8, gt=0)

    model_config = ConfigDict(extra="ignore")

class OptimizationRequest(BaseModel):
    session_id: str = Field(..., description="Unique session ID to persist optimizer state (momentum, etc.)")
    config: OptimizerConfig
    parameters: List[float] = Field(..., description="Flattened list of parameters")
    gradients: List[float] = Field(..., description="Flattened list of gradients")

class OptimizationResponse(BaseModel):
    updated_parameters: List[float]
    status: str
    message: Optional[str] = None

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class QPFInput(BaseModel):
    objective: str = Field(..., description="Objective: e.g., Develop a mean-reversion strategy")
    universe: str = Field(..., description="Universe: e.g., S&P 500 tech stocks")
    data_frequency: str = Field(..., description="Data Frequency: e.g., 1-hour bars")
    methodology: str = Field(..., description="Methodology: e.g., Cointegration and Bollinger Bands")
    deliverable: str = Field(..., description="Deliverable: e.g., Python code using VectorBT")
    risk_metrics: str = Field(..., description="Risk Metrics: e.g., Sharpe Ratio and Max Drawdown")
    constraints: str = Field(..., description="Constraints: e.g., 0.1% slippage and $10k starting capital")

class QPFOutput(BaseModel):
    python_code: str = Field(..., description="Modular Python code")
    performance_report: str = Field(..., description="Report with risk metrics")
    quant_critique: str = Field(..., description="Critique identifying flaws like look-ahead bias")

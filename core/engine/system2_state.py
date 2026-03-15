from typing import TypedDict, Dict, Any, List, Optional
from pydantic import BaseModel, Field

# --- Pydantic Models for DCF Constraints & Generation ---

class DCFAssumptions(BaseModel):
    revenue_growth_rate: float = Field(..., description="Projected annualized revenue growth rate (e.g., 0.05 for 5%)")
    operating_margin: float = Field(..., description="Projected operating margin (e.g., 0.20 for 20%)")
    tax_rate: float = Field(..., description="Assumed effective tax rate (e.g., 0.21 for 21%)")
    capital_expenditure_margin: float = Field(..., description="CapEx as a percentage of revenue")
    depreciation_margin: float = Field(..., description="D&A as a percentage of revenue")
    change_in_nwc_margin: float = Field(..., description="Change in Net Working Capital as a percentage of revenue")

class DCFModelOutput(BaseModel):
    company_ticker: str = Field(..., description="The stock ticker symbol")
    wacc: float = Field(..., description="Weighted Average Cost of Capital (Discount Rate) (e.g., 0.10 for 10%)")
    terminal_growth_rate: float = Field(..., description="Perpetual growth rate used in Terminal Value calculation")
    assumptions: DCFAssumptions = Field(..., description="Underlying assumptions driving the free cash flows")
    projected_fcfs: List[float] = Field(..., description="List of projected Free Cash Flows for the forecast period (e.g., 5 years)")
    terminal_value: float = Field(..., description="Calculated Terminal Value at the end of the forecast period")
    enterprise_value: float = Field(..., description="The calculated total Enterprise Value (NPV of FCFs + NPV of TV)")
    implied_share_price: Optional[float] = Field(None, description="The implied share price after adjusting for net debt and shares outstanding")

# --- LangGraph State Definition ---

class System2State(TypedDict):
    """
    The state structure for the System 2 LangGraph workflow.
    Controls the Reflexion loop for financial modeling.
    """
    company_ticker: str                  # The ticker being analyzed
    historical_data: Dict[str, Any]      # Extracted historical financials/context
    
    # Execution & Validation State
    iteration_count: int                 # Number of times the loop has run
    max_iterations: int                  # Maximum allowed reflexion loops
    
    # DCF Generation State
    generated_dcf: Optional[Dict[str, Any]] # Raw dict representation of DCFModelOutput
    validation_feedback: List[str]       # Feedback from the Reflector node if rules are violated
    is_valid: bool                       # Flag set by the Reflector node
    
    # Final Output
    final_report: str                    # The synthesized narrative report

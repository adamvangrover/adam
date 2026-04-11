from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any

class MarketState(BaseModel):
    asset_ticker: str = Field(..., description="The ticker or CUSIP of the asset.")
    asset_profile: str = Field(..., description="Profile of the asset, e.g., Distressed Debt.")
    days_since_last_print: int = Field(..., description="Number of days since the asset last traded.")
    current_inventory: int = Field(..., description="Current long/short position size in units.")
    macro_context: str = Field(..., description="Relevant macroeconomic variables or constraints.")

class MarketQuote(BaseModel):
    bid: float = Field(..., description="The generated bid price.")
    ask: float = Field(..., description="The generated ask price.")
    justification: str = Field(..., description="Agent's reasoning for the spread and skew.")
    inventory_skew_applied: bool = Field(..., description="Whether a skew was applied due to inventory limits.")

    @model_validator(mode='after')
    def check_spread(self) -> 'MarketQuote':
        if self.bid >= self.ask:
            raise ValueError("Bid price must be less than ask price.")
        return self

class LlmGrade(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Score from 1 to 5.")
    feedback: str = Field(..., description="Qualitative feedback on the reasoning.")
    missed_risk_factors: List[str] = Field(default_factory=list, description="Risk factors the agent missed.")

class EvalResultMetrics(BaseModel):
    spread: float = Field(..., description="The absolute spread (ask - bid).")
    spread_bps: Optional[float] = Field(None, description="The spread in basis points relative to fair value.")

class EvalResult(BaseModel):
    passed_deterministic: bool = Field(..., description="Whether the quote passed all hard logic checks.")
    deterministic_errors: List[str] = Field(default_factory=list, description="List of hard logic errors encountered.")
    metrics: EvalResultMetrics = Field(..., description="Metrics computed from the quote.")
    qualitative_grade: Optional[LlmGrade] = Field(None, description="The grade from the LLM Judge if applicable.")

if __name__ == "__main__":
    # Test instantiation
    quote = MarketQuote(bid=82.5, ask=91.0, justification="Distressed asset.", inventory_skew_applied=True)
    print(quote.model_dump_json(indent=2))

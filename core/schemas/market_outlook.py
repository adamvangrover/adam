from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class ConvictionName(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol.")
    company_name: str = Field(..., description="The full company name.")
    conviction_level: str = Field(..., description="Level of conviction (e.g., High, Medium).")
    thesis: str = Field(..., description="The core investment thesis for this name.")
    target_price: Optional[float] = Field(None, description="The 12-month target price.")
    key_catalysts: List[str] = Field(default_factory=list, description="List of upcoming catalysts.")

class MarketPrediction(BaseModel):
    timeframe: str = Field(..., description="The timeframe of the prediction (e.g., Q3 2026, Next 12 Months).")
    asset_class: str = Field(..., description="The asset class (e.g., Equities, Fixed Income, Commodities).")
    prediction_summary: str = Field(..., description="A short summary of the prediction.")
    probability: float = Field(..., description="Probability of occurrence (0.0 to 1.0).")
    rationale: str = Field(..., description="Detailed rationale supporting the prediction.")

    @field_validator('probability')
    def validate_probability(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Probability must be between 0.0 and 1.0')
        return v

class MarketMayhemOverview(BaseModel):
    date: str = Field(..., description="Date of the overview in YYYY-MM-DD format.")
    headline: str = Field(..., description="Main headline for the market overview.")
    volatility_index: float = Field(..., description="The VIX or similar volatility metric.")
    market_sentiment: str = Field(..., description="Overall sentiment (e.g., Bullish, Bearish, Neutral, Panic).")
    key_drivers: List[str] = Field(default_factory=list, description="List of the primary drivers of market action today.")
    narrative_summary: str = Field(..., description="A detailed paragraph summarizing the 'Market Mayhem'.")

class DailyOutlookReport(BaseModel):
    overview: MarketMayhemOverview = Field(..., description="The high-level market mayhem overview.")
    predictions: List[MarketPrediction] = Field(default_factory=list, description="Key market predictions.")
    top_ten_convictions: List[ConvictionName] = Field(default_factory=list, description="Top ten conviction name deep dives.", max_length=10)

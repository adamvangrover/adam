from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class FinancialMetrics(BaseModel):
    revenue: float
    ebitda: float
    net_income: float
    total_debt: float
    cash: float
    net_debt: Optional[float] = None
    leverage_ratio: Optional[float] = None

class RAGSource(BaseModel):
    doc_id: str
    chunk_id: str
    page_number: int

class CreditMemoSection(BaseModel):
    title: str
    content: str
    citations: List[RAGSource]

class CreditMemo(BaseModel):
    borrower_name: str
    ticker: str
    sector: str
    report_date: datetime
    risk_score: float
    executive_summary: str
    sections: List[CreditMemoSection]
    historical_financials: List[Dict[str, Any]]
    dcf_analysis: Dict[str, Any]
    pd_model: Dict[str, Any]

class Report(BaseModel):
    id: str
    title: str
    date: str
    type: str # "DAILY_BRIEFING" or "MARKET_PULSE"
    content_html: str
    metadata: Dict[str, Any]

class Agent(BaseModel):
    name: str
    role: str
    description: str
    tools: List[str]
    model: str = "gpt-4"

class Prompt(BaseModel):
    name: str
    template: str
    variables: List[str]

class CrisisScenario(BaseModel):
    name: str
    duration: int
    market_volatility: float
    market_trend: float
    shock_event_day: int
    shock_magnitude: float

class SimulationResult(BaseModel):
    scenario: str
    labels: List[int]
    market: List[float]
    portfolio: List[float]

class SystemMetric(BaseModel):
    name: str
    value: float
    unit: str
    timestamp: datetime

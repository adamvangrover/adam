import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# --- Data Schemas (Pydantic) ---

class BalanceSheet(BaseModel):
    """Structured Balance Sheet Data."""
    cash_equivalents: float = Field(..., description="Cash and Cash Equivalents")
    total_assets: float = Field(..., description="Total Assets")
    total_debt: float = Field(..., description="Total Debt including short and long term")
    equity: float = Field(..., description="Total Shareholders Equity")
    currency: str = Field("USD", description="Reporting currency")
    fiscal_year: int = Field(..., description="Fiscal Year")

class IncomeStatement(BaseModel):
    """Structured Income Statement Data."""
    revenue: float = Field(..., description="Total Revenue")
    operating_income: float = Field(..., description="Operating Income")
    net_income: float = Field(..., description="Net Income")
    depreciation_amortization: float = Field(..., description="D&A")
    interest_expense: float = Field(..., description="Interest Expense")
    consolidated_ebitda: Optional[float] = Field(None, description="Calculated EBITDA")

class CovenantDefinition(BaseModel):
    """Legal definition of a covenant."""
    name: str = Field(..., description="Name of the covenant, e.g., 'Net Leverage Ratio'")
    threshold: float = Field(..., description="The limit value")
    operator: Literal[">", "<", ">=", "<="] = Field(..., description="Comparison operator")
    definition_text: str = Field(..., description="Exact legal text defining the covenant")
    add_backs: List[str] = Field(default_factory=list, description="List of permitted add-backs")

class RiskFactor(BaseModel):
    """Defines a single dimension of the market risk hypercube."""
    name: str
    current_value: float
    volatility: float  # Annualized volatility
    mean_reversion: float = 0.0  # Ornstein-Uhlenbeck parameter (kappa)

class MarketScenario(BaseModel):
    """
    Represents a generated market scenario for stress testing.
    A single point in the high-dimensional risk manifold.
    """
    scenario_id: str
    description: str
    # Core macroeconomic indicators
    risk_factors: Dict[str, float] = Field(..., description="Key risk indicators (e.g., 'inflation', 'unemployment')")
    # Meta-data
    probability_weight: float = Field(1.0, description="The likelihood of this scenario occurring relative to the batch")
    is_tail_event: bool = Field(False, description="Whether this represents a statistical outlier (>3 sigma)")
    regime_label: str = Field("normal", description="The market regime this scenario belongs to")

class InvestmentMemo(BaseModel):
    """The Final Output."""
    executive_summary: str
    key_risks: List[str]
    financial_analysis: str
    legal_analysis: str
    recommendation: Literal["BUY", "SELL", "HOLD"]
    confidence_score: float

# --- Graph State (TypedDict) ---

class VerticalRiskGraphState(TypedDict):
    """
    State for the Credit Risk Vertical AI Agent.
    """
    # Inputs
    ticker: str
    data_room_path: str

    # Extracted Data (Pydantic objects serialized or dicts)
    balance_sheet: Optional[Dict[str, Any]]
    income_statement: Optional[Dict[str, Any]]
    covenants: List[Dict[str, Any]]

    # Analysis
    quant_analysis: Optional[str]
    legal_analysis: Optional[str]
    market_research: Optional[str]
    
    # --- Merged Simulation State ---
    # Stores the definitions of scenarios (inputs to the engine)
    stress_scenarios: Optional[List[Dict[str, Any]]] 
    
    # Stores the quantitative results of the simulation (outputs from the engine)
    risk_simulation_results: Optional[List[Dict[str, Any]]]

    # Draft
    draft_memo: Optional[Dict[str, Any]] # Serialized InvestmentMemo

    # Control Flow
    messages: Annotated[List[Any], operator.add] # Chat history for the Supervisor
    next_step: Optional[str]
    critique_count: int
    human_feedback: Optional[str]

    # Explainability
    status: str
from typing import TypedDict, List, Optional, Annotated, Dict, Any, Literal
import operator
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
    # v23.5 Additions
    risk_simulation_results: Optional[List[Dict[str, Any]]]  # Output from StochasticRiskEngine (Merton/Cholesky)

    # Draft
    draft_memo: Optional[Dict[str, Any]] # Serialized InvestmentMemo

    # Control Flow
    messages: Annotated[List[Any], operator.add] # Chat history for the Supervisor
    next_step: Optional[str]
    critique_count: int
    human_feedback: Optional[str]

    # Explainability
    status: str

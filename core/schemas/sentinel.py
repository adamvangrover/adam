from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union

class CreditMetrics(BaseModel):
    pd: float = Field(..., description="Probabilistic Probability of Default")
    lgd: float = Field(..., description="Deterministic Loss Given Default")
    ead: float = Field(..., description="Exposure at Default")
    expected_loss: float = Field(0.0, description="Calculated Expected Loss")

    @model_validator(mode='after')
    def calculate_expected_loss(self):
        self.expected_loss = self.pd * self.lgd * self.ead
        return self

class DecisionState(BaseModel):
    conviction_score: float = Field(ge=0, le=1)
    routing_path: str # ["AUTOMATED", "HOTL", "HITL_TIER_3"]
    requires_step_up: bool
    audit_hash: str # SHA-256 of the context + prompt

class IncomeStatement(BaseModel):
    revenue: Optional[float] = None
    cogs: Optional[float] = None
    ebitda_adjusted: Optional[float] = None

class BalanceSheet(BaseModel):
    cash_and_equivalents: Optional[float] = None
    total_senior_debt: Optional[float] = None
    total_subordinated_debt: Optional[float] = None

class CashFlow(BaseModel):
    operating_cash_flow: Optional[float] = None
    capex_maintenance: Optional[float] = None
    free_cash_flow: Optional[float] = None

class ReportingPeriod(BaseModel):
    period_end_date: Optional[str] = None
    period_type: Optional[str] = None # ["Annual", "Quarterly", "LTM"]
    income_statement: Optional[IncomeStatement] = None
    balance_sheet: Optional[BalanceSheet] = None
    cash_flow: Optional[CashFlow] = None

class FinancialSpreadTemplate(BaseModel):
    entity_id: Optional[str] = None
    reporting_periods: Optional[List[ReportingPeriod]] = None

class CapitalAccountAnalysis(BaseModel):
    beginning_capital_account: Optional[float] = None
    capital_contributed: Optional[float] = None
    current_year_net_income: Optional[float] = None
    withdrawals_distributions: Optional[float] = None
    ending_capital_account: float

class PartnerShareOfLiabilities(BaseModel):
    nonrecourse: Optional[float] = None
    qualified_nonrecourse_financing: Optional[float] = None
    recourse: Optional[float] = None

class ScheduleK1Extraction(BaseModel):
    partner_identifying_number: Optional[str] = Field(None, pattern=r"^[0-9]{9}$")
    partnership_name: Optional[str] = None
    capital_account_analysis: Optional[CapitalAccountAnalysis] = None
    partner_share_of_liabilities: Optional[PartnerShareOfLiabilities] = None

class Covenant(BaseModel):
    covenant_type: Optional[str] = None # ["Maintenance", "Incurrence"]
    metric_name: Optional[str] = None # ["Total Leverage Ratio", "Fixed Charge Coverage Ratio", "Minimum Liquidity"]
    threshold_value: Optional[float] = None
    threshold_operator: Optional[str] = None # ["<=", ">=", "=="]
    current_calculated_value: Optional[float] = None
    compliance_status: Optional[str] = None # ["Pass", "Warning", "Breach"]
    cure_period_days: Optional[int] = None

class CovenantComplianceMonitor(BaseModel):
    credit_agreement_id: Optional[str] = None
    covenants: Optional[List[Covenant]] = None

class RegulatoryNarrative(BaseModel):
    primary_repayment_source_analysis: Optional[str] = None
    secondary_repayment_source_analysis: Optional[str] = None
    mitigating_factors: Optional[str] = None
    examiner_conclusion: Optional[str] = None

class RegulatoryClassificationReport(BaseModel):
    borrower_name: Optional[str] = None
    snc_rating: Optional[str] = None # ["Pass", "Special Mention", "Substandard", "Doubtful", "Loss"]
    regulatory_narrative: Optional[RegulatoryNarrative] = None
    date_of_exam: Optional[str] = None

class QuantitativeFactors(BaseModel):
    merton_distance_to_default: Optional[float] = None
    altman_z_score: Optional[float] = None
    debt_service_coverage_ratio_ttm: Optional[float] = None

class QualitativeFactors(BaseModel):
    management_experience_score: Optional[float] = Field(None, ge=1, le=5)
    industry_headwinds_indicator: Optional[str] = None # ["Low", "Moderate", "Severe"]
    sponsor_support_strength: Optional[str] = None # ["Strong", "Adequate", "Weak"]

class ProbabilityOfDefaultModel(BaseModel):
    composite_pd_percentage: Optional[float] = Field(None, ge=0, le=100)
    implied_sp_rating: Optional[str] = None # ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
    quantitative_factors: Optional[QuantitativeFactors] = None
    qualitative_factors: Optional[QualitativeFactors] = None

class DebtTranche(BaseModel):
    tranche_name: Optional[str] = None
    seniority_rank: Optional[int] = None
    outstanding_principal: Optional[float] = None
    allocated_recovery_value: Optional[float] = None
    recovery_percentage: Optional[float] = Field(None, ge=0, le=100)

class CapitalStructureRecovery(BaseModel):
    enterprise_valuation_scenario: Optional[str] = None # ["Base Case", "Stress Case", "Liquidation"]
    total_distributable_value: Optional[float] = None
    debt_tranches: Optional[List[DebtTranche]] = None
    equity_residual_value: Optional[float] = None

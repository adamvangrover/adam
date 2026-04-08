from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union, Literal

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


class TransactionMetadata(BaseModel):
    transaction_id: str
    inference_timestamp: str
    execution_modality: Literal["AUTOMATED", "HOTL", "HITL_TIER_3"]


class AuditMetadata(BaseModel):
    model_registry_id: str
    model_version_hash: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    data_snapshot_uri: str
    pipeline_execution_id: Optional[str] = None


class PredictionMetrics(BaseModel):
    probability_of_default: float = Field(..., ge=0, le=1)
    decision_threshold: float
    algorithmic_decision: Literal["APPROVE", "DENY", "ESCALATE"]
    confidence_interval: Optional[List[float]] = Field(None, min_length=2, max_length=2)


class FeatureAttribution(BaseModel):
    feature_name: str
    raw_feature_value: float
    shap_marginal_contribution: float


class ExplainabilityPayload(BaseModel):
    xai_methodology: Literal["SHAP", "LIME"]
    base_value: float
    feature_attributions: List[FeatureAttribution]


class CreditDecision_XAIPayload(BaseModel):
    transaction_metadata: TransactionMetadata
    audit_metadata: AuditMetadata = Field(..., alias="_audit_metadata")
    prediction_metrics: PredictionMetrics
    explainability_payload: ExplainabilityPayload


class PrincipalDenialReason(BaseModel):
    reason_rank: int = Field(..., ge=1, le=4)
    form_c1_code: Literal["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    consumer_facing_narrative: str
    aggregated_shap_weight: float
    constituent_features: Optional[List[str]] = None


class AuditValidation(BaseModel):
    xai_fidelity_score: Optional[float] = Field(None, ge=0, le=1)
    requires_human_review: Optional[bool] = None
    mapping_algorithm_version: Optional[str] = None


class AdverseAction_RegB_Translation(BaseModel):
    transaction_id: str
    regulatory_framework: Literal["ECOA_Regulation_B"] = "ECOA_Regulation_B"
    principal_denial_reasons: List[PrincipalDenialReason] = Field(..., max_length=4)
    audit_validation: Optional[AuditValidation] = None

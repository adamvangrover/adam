"""
Module 8: Hunt & Kill — Mercenary Arbitrage & Predatory Execution (Hardened Spec)
Pydantic Data Contracts & Schemas for ADAM v30.1-R-Patch Extension.
Enforces Lexicographic Control Barrier Functions (CBFs) & Safety Primacy (L0 > L1 > L2).
"""

from typing import Literal

from pydantic import BaseModel, Field

EpistemicStatus = Literal[
    "[VERIFIED]",
    "[CALCULATED]",
    "[MODEL_EST]",
    "[PENDING]",
    "[SURVEILLANCE_RULE_SPEC]",
    "[ADVERSARIAL_NOISE_QUARANTINE]"
]

DislocationVector = Literal[
    "Vector 1: Capital Structure & LMT Priming Arbitrage",
    "Vector 2: Cash vs. Synthetic Credit Basis Compression",
    "Vector 3: Dispersion & Correlation Arbitrage",
    "Vector 4: ETF Creation/Redemption & NAV Siphoning",
    "Vector 5: Counter-Swarm Liquidity Ambush"
]

TriageCategory = Literal[
    "Adversarial Swarm Liquidity Traps",
    "Rating-Cliff Fallen Angels",
    "Vol-Target & Risk-Parity De-Grossing",
    "Maturity Wall & Covenant Traps"
]

TargetLifecycleState = Literal[
    "LOCATED",
    "ISOLATED_PASS",
    "ISOLATED_QUARANTINED",
    "EXPLOITED",
    "EXTINGUISHING",
    "SWEPT_TO_FORTRESS",
    "KILL_SWITCH_FLATTENED"
]


class Layer0SafetyHeader(BaseModel):
    """
    Layer 0 Multiplicative Control Barrier Function (CBF) Safety Gate Header.
    Final Assured Score = Layer0_Gate (0 or 1) * Layer1_Quality_Score.
    """
    layer_0_gate: int = Field(..., ge=0, le=1, description="Binary multiplier: 1 = PASS, 0 = HARD_ABORT")
    h1_vpin_toxicity_pass: bool = Field(..., description="h1 = 0.65 - VPIN >= 0 (and SNER <= P99)")
    h2_short_utilization_pass: bool = Field(..., description="h2 = 85.0% - Util_Short >= 0")
    h3_borrow_fee_pass: bool = Field(..., description="h3 = 450 bps - Fee_Borrow >= 0")
    h4_sacred_rights_coop_pass: bool = Field(..., description="h4 = Stake_CoOp - Threshold_SacredRights >= 0")
    sha256_feed_integrity_verified: bool = Field(default=True, description="SHA-256 schema & feed integrity check")
    epistemic_confidence_score: float = Field(..., ge=0.0, le=1.0, description="Epistemic confidence score (>= 0.85 required)")
    hitl_escalation_required: bool = Field(..., description="True if confidence < 0.85 or Assured Score < 4.00")
    final_assured_score: float = Field(..., description="Layer 0 * Layer 1 Quality Rubric score")
    market_integrity_compliance_pass: bool = Field(default=True, description="Strict compliance with MAR & Rule 10b-5 (no spoofing/MNPI)")


class ForcedFlowTriageMetrics(BaseModel):
    """Stage 1 (Locate) mechanical balance-sheet & swarm constraint quantitative metrics."""
    downgrade_probability_pct: float = Field(..., description="Rating-cliff downgrade probability to BB+")
    realized_vol_10d: float = Field(..., description="10-day realized volatility")
    vix_anchor_ratio: float = Field(..., description="Realized Vol 10D / 18.62 VIX Anchor ratio")
    forced_de_grossing_selling_pressure_mm: float = Field(..., description="Estimated mechanical selling pressure ($M)")
    ltm_interest_coverage: float = Field(..., description="Trailing 12-month Interest Coverage ratio")
    ltm_fccr: float = Field(..., description="Trailing 12-month Fixed Charge Coverage ratio")
    covenant_trap_active: bool = Field(..., description="True if Interest Coverage < 1.50x or FCCR < 1.00x")
    vpin: float = Field(..., ge=0.0, le=1.0, description="Volume-Synchronized Probability of Informed/Swarm Trading")
    sner: float = Field(..., description="Signal-to-Noise Exploit Ratio (Cancel/Fill ratio change / Settlement volume change)")
    swarm_trap_detected: bool = Field(..., description="True if VPIN > 0.65 and SNER exceeds 99th historical percentile")


class MarketImpactSlippage(BaseModel):
    """Square-root market impact algorithm slippage in bps."""
    position_size_dollars: float = Field(..., description="Proposed position size in USD")
    average_daily_volume_dollars: float = Field(..., description="Average Daily Volume (ADV) in USD")
    participation_rate_pct: float = Field(..., description="Participation rate percentage of ADV")
    participation_gamma: float = Field(default=0.75, description="Gamma market impact parameter")
    calculated_slippage_bps: float = Field(..., description="Calculated ADV-based market impact slippage in bps")
    swarm_penalty_bps: float = Field(default=0.0, description="Toxicity penalty subtracted due to VPIN / SNER")


class EmbeddedSafetyDefense(BaseModel):
    """
    Embedded Safety Invariant Check enforcing Safety Primacy.
    Guarantees expected return / spread / PCS can NEVER override safety constraints.
    """
    target_id: str | None = Field(None, description="Target ID")
    safety_primacy_enforced: bool = Field(
        default=True,
        description="True invariant: Safety constraints strictly take precedence over return/PCS"
    )
    short_squeeze_ceiling_breached: bool = Field(
        ...,
        description="True if utilization > 85.0% or borrow cost > 450 bps"
    )
    coop_legality_breached: bool = Field(
        ...,
        description="True if LMT Priming vector lacks verified blocking position or co-op agreement"
    )
    vpin_toxicity_quarantine_triggered: bool = Field(
        ...,
        description="True if VPIN > 0.65 and SNER > P99 without primary clearing print"
    )
    unbounded_loss_risk_detected: bool = Field(
        ...,
        description="True if position exhibits unhedged gap risk or tail risk violating portfolio VaR"
    )
    reward_override_attempted: bool = Field(
        ...,
        description="True if trade met financial criteria (PCS >= 2.50) but was blocked due to safety breach"
    )
    safety_verdict: Literal["PASS", "HARD_ABORT"] = Field(
        ...,
        description="HARD_ABORT if any safety invariant is breached, regardless of financial reward"
    )


class PCSCalculation(BaseModel):
    """Toxicity-Adjusted Predatory Conviction Score (PCS) calculation details."""
    expected_convergence_bps: float = Field(..., description="Expected convergence in basis points")
    probability_catalyst_le_t: float = Field(..., ge=0.0, le=1.0, description="Probability that catalyst occurs within window T_max")
    vpin_toxicity: float = Field(..., ge=0.0, le=1.0, description="VPIN order-flow toxicity score")
    max_adverse_excursion_bps: float = Field(..., gt=0.0, description="Maximum Adverse Excursion in basis points")
    illiquidity_haircut: float = Field(..., gt=0.0, description="Illiquidity haircut factor (>= 1.0)")
    pcs_score: float = Field(
        ...,
        description="Calculated Toxicity-Adjusted PCS = (Expected Convergence * Pr(Catalyst <= T) * (1 - VPIN)) / (MAE * Illiquidity Haircut)"
    )
    authorized: bool = Field(..., description="True if pcs_score >= 2.50")


class DislocationTarget(BaseModel):
    """Target dislocated entity or security pair identified in Stage 1 (Locate)."""
    target_id: str = Field(..., description="Unique target ID (e.g. HK-01)")
    target_name: str = Field(..., description="Target name or instrument pair")
    triage_category: TriageCategory = Field(..., description="Forced-flow triage category")
    dislocation_vector: DislocationVector = Field(..., description="Predatory playbook vector")
    long_leg_instrument: str = Field(..., description="Long leg instrument name")
    long_leg_price: float = Field(..., description="Long leg spot price")
    short_leg_instrument: str = Field(..., description="Short/hedge leg instrument name")
    short_leg_price: float = Field(..., description="Short/hedge leg spot price")
    implied_pricing: float = Field(..., description="Model implied fair pricing")
    spot_pricing: float = Field(..., description="Observed market spot pricing")
    cost_repo_sofr_bps: float = Field(..., description="Annualized Repo/SOFR funding cost in bps")
    bid_ask_slippage_bps: float = Field(..., description="Estimated bid-ask slippage in bps")
    carry_theta_div_bps: float = Field(..., description="Net Theta/Dividend carry cost in bps")
    catalyst_description: str = Field(..., description="Expected convergence catalyst")
    t_max_days: int = Field(..., gt=0, description="Maximum catalyst window in trading days")
    position_size_usd: float = Field(default=10_000_000.0, description="Proposed position size in USD")
    adv_usd: float = Field(default=50_000_000.0, description="Average Daily Volume in USD")
    vpin: float = Field(default=0.25, ge=0.0, le=1.0, description="Order-flow toxicity VPIN")
    sner: float = Field(default=1.2, description="Signal-to-Noise Exploit Ratio")


class OrderRoutingSpec(BaseModel):
    """Order routing and execution microstructure spec for Stage 3 (Exploit)."""
    target_id: str
    air_gapped_sanitization_verified: bool = Field(default=True, description="True if unstructured strings stripped prior to execution")
    liquidity_masking_venue: str = Field(default="Dark pools / midpoint pegs (IEX D-Limit)", description="Routing venue")
    execution_algorithm: str = Field(default="Implementation Shortfall (IS) with randomized child order arrival", description="Execution algorithm")
    credit_derivative_legging_sequence: list[str] = Field(
        default_factory=lambda: [
            "1. Execute physical leg (TRACE / 1L BSL assignment / basket) first",
            "2. Deploy contingent delta-hedge in liquid futures / CDX NA HY",
            "3. Lift synthetic leg to eliminate legging risk"
        ],
        description="Legging sequence"
    )


class CounterPredationCheck(BaseModel):
    """Counter-Predation telemetry and safety checks."""
    target_id: str
    short_squeeze_utilization_pct: float = Field(..., description="Short borrow utilization percentage")
    stock_loan_borrow_cost_bps: float = Field(..., description="Annual stock-loan borrow cost in bps")
    borrow_recall_ceiling_pass: bool = Field(..., description="True if utilization <= 85% and borrow cost <= 450 bps")
    coop_steering_position_pct: float | None = Field(None, description="Steering group creditor holding percentage")
    coop_majority_verified: bool = Field(..., description="True if blocking position (>33.4% or >50.1%) or signed coop agreement verified")
    overall_pass: bool = Field(..., description="True if all counter-predation checks pass")


class BasisLedgerEntry(BaseModel):
    """Standardized institutional entry in the Structural Dislocation & Basis Ledger."""
    kill_id: str = Field(..., description="Kill ID (e.g., HK-01)")
    target_pair: str = Field(..., description="Target / Pair name")
    dislocation_vector: DislocationVector = Field(..., description="Dislocation Vector")
    long_leg: str = Field(..., description="Long Leg (Instrument / Price)")
    short_leg: str = Field(..., description="Short / Hedge Leg (Instrument / Price)")
    gross_dislocation_bps: float = Field(..., description="Gross Dislocation in bps or %")
    funding_slippage_cost_bps: float = Field(..., description="Total Funding, Slippage & Swarm Penalty in bps")
    net_exploitable_edge_bps: float = Field(..., description="Net Exploitable Edge (Spread_Net) in bps")
    vpin_sner: str = Field(..., description="VPIN / SNER toxicity ratio")
    pcs_score: float = Field(..., description="Toxicity-Adjusted Predatory Conviction Score")
    epistemic_status: EpistemicStatus = Field(..., description="Epistemic status tag")
    lifecycle_state: TargetLifecycleState = Field(..., description="Target lifecycle state")
    convergence_catalyst_and_tmax: str = Field(..., description="Convergence Catalyst & Max Window (T_max)")


class ExtinguishTelemetry(BaseModel):
    """Stage 4 (Extinguish) basis convergence and multi-oracle kill-switch telemetry."""
    target_id: str
    initial_spread_bps: float
    current_spread_bps: float
    compression_pct: float = Field(..., description="Percentage of dislocation compressed")
    scale_out_50_triggered: bool = Field(..., description="True if compression >= 70%")
    full_extinguish_triggered: bool = Field(..., description="True if remaining spread <= marginal SOFR funding cost")
    multi_oracle_180s_consensus_verified: bool = Field(default=True, description="Verified across >= 2 distinct asset classes over 180s")
    t_max_expired: bool = Field(..., description="True if catalyst window T_max expired")
    repo_haircut_expanded_gt_150bps: bool = Field(..., description="True if repo haircuts expanded by >150 bps")
    z_score_excursion_gt_2: bool = Field(..., description="True if cross-asset |z| > 2.0")
    hard_kill_switch_active: bool = Field(..., description="True if any adverse regime kill-switch is triggered")
    capital_sweep_to_sovereign_fortress: bool = Field(..., description="True if realized profits/principal swept to Sovereign Bill Fortress")


class HuntAndKillInput(BaseModel):
    """Input contract for the Hunt & Kill execution skill."""
    correlation_id: str | None = Field(None, description="Unique correlation ID")
    sofr_base_rate_pct: float = Field(default=3.88, description="SOFR benchmark rate (%) [VERIFIED NY Fed Oct 2026]")
    hurdle_rate_bps: float = Field(default=150.0, description="Annualized hurdle rate over financing (bps)")
    targets: list[DislocationTarget] = Field(..., description="List of target dislocation candidates")
    counter_predation_inputs: list[CounterPredationCheck] = Field(..., description="Counter-predation inputs per target")


class HuntAndKillOutput(BaseModel):
    """Output contract for the Hunt & Kill execution skill."""
    correlation_id: str
    timestamp: str
    safety_header: Layer0SafetyHeader = Field(..., description="Layer 0 Safety & Materiality Assurance Header")
    triage_metrics: list[ForcedFlowTriageMetrics] = Field(default_factory=list, description="Quantitative forced-flow triage metrics")
    basis_ledger: list[BasisLedgerEntry] = Field(..., description="Structural Dislocation & Basis Ledger")
    pcs_calculations: list[PCSCalculation] = Field(..., description="PCS Score breakdown per target")
    market_impacts: list[MarketImpactSlippage] = Field(default_factory=list, description="ADV-based market impact calculations")
    order_routing_specs: list[OrderRoutingSpec] = Field(..., description="Order routing microstructure specs")
    counter_predation_audit: list[CounterPredationCheck] = Field(..., description="Counter-predation verification ledger")
    safety_defenses: list[EmbeddedSafetyDefense] = Field(..., description="Embedded Safety Primacy Invariant audit ledger")
    extinguish_telemetry: list[ExtinguishTelemetry] = Field(..., description="Basis convergence & Kill-switch telemetry")
    total_authorized_deployments: int
    total_quarantined_deployments: int
    capital_swept_to_fortress_ballast_pct: float = Field(default=100.0, description="Profits swept back to Sovereign Bill Fortress Ballast")
    status: str = Field(default="COMPLETED")

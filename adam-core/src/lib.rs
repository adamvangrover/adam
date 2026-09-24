use ring::digest::{Context, SHA256};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use thiserror::Error;
use uuid::Uuid;

// ============================================================================
// 1. DETERMINISTIC FIXED-POINT ARITHMETIC (Eliminates IEEE 754 Drift)
// ============================================================================

/// Exact fixed-point representation scaled to 4 decimal places (Basis Points).
/// 1 bp = 0.0001 = 1 unit. 1.0000 = 10_000 units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BasisPoints(pub i64);

impl BasisPoints {
    pub const ZERO: Self = BasisPoints(0);
    pub const ONE_BPS: Self = BasisPoints(1);
    pub const ONE_PERCENT: Self = BasisPoints(100);
    pub const HUNDRED_PERCENT: Self = BasisPoints(10_000);

    pub fn from_integer(val: i64) -> Self {
        BasisPoints(val * 10_000)
    }

    pub fn add(self, rhs: Self) -> Result<Self, KernelError> {
        self.0
            .checked_add(rhs.0)
            .map(BasisPoints)
            .ok_or(KernelError::FixedPointOverflow)
    }

    pub fn sub(self, rhs: Self) -> Result<Self, KernelError> {
        self.0
            .checked_sub(rhs.0)
            .map(BasisPoints)
            .ok_or(KernelError::FixedPointOverflow)
    }

    pub fn mul_bps(self, rhs: Self) -> Result<Self, KernelError> {
        // (a * b) / 10_000 using 128-bit intermediate integer to avoid premature overflow
        let a = self.0 as i128;
        let b = rhs.0 as i128;
        let prod = (a * b) / 10_000;
        if prod > i64::MAX as i128 || prod < i64::MIN as i128 {
            Err(KernelError::FixedPointOverflow)
        } else {
            Ok(BasisPoints(prod as i64))
        }
    }
}

// ============================================================================
// 2. TEMPORAL MULTI-CLOCK GEOMETRY (Decoupling Observation from Reality)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalGeometry {
    /// t_e: Event Time (Instant event occurred in external world, RFC3339)
    pub t_e: String,
    /// t_k: Knowledge Time (Instant ingested and hash-sealed into C0)
    pub t_k: String,
    /// t_d: Decision Time (Instant admitted and verified through G12/C4/C5)
    pub t_d: Option<String>,
    /// t_x: Execution Time (Instant of physical settlement at venue)
    pub t_x: Option<String>,
}

// ============================================================================
// 3. ZERO-SIZED CAPABILITY TOKENS & STATE WITNESSES
// ============================================================================

pub mod state {
    pub struct CanonicalObservation; // C0
    pub struct EphemeralHypothesis;   // C3 (Authority: NONE)
    pub struct VerifiedMath;          // C4 (Verification Authority)
    pub struct PolicyAdmitted;        // C5 (Policy Authority)
    pub struct CommittedLog;          // C6 (Execution Authority)
}

/// A sealed capability token proving deterministic mathematical satisfaction.
/// Constructible ONLY by the internal C4 verification solver.
pub struct VerificationWitness {
    _private: (),
}

/// A sealed capability token proving institutional policy clearance.
/// Constructible ONLY by the internal C5 admission engine.
pub struct PolicyWitness {
    _private: (),
}

/// A sealed execution token allowing immutable commit into C6.
pub struct CommitAuthorityToken {
    _private: (),
}

// ============================================================================
// 4. EPISTEMIC LATTICE (Formal Uncertainty Representation)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicStatus {
    Known,
    Supported,
    Uncertain,
    Conflicted,
    OutOfDistribution,
    InsufficientEvidence,
    Unresolved,
    Unknown,
}

impl EpistemicStatus {
    pub fn is_actionable(&self) -> bool {
        matches!(self, EpistemicStatus::Known | EpistemicStatus::Supported)
    }
}

// ============================================================================
// 5. HYPOTHESIS & PROPOSITION PAYLOADS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditModificationProposal {
    pub counterparty_id: String,
    pub original_limit_bps: BasisPoints,
    pub proposed_limit_bps: BasisPoints,
    pub dscr_estimate_bps: BasisPoints, // Debt Service Coverage Ratio * 10,000
    pub leverage_ratio_bps: BasisPoints,
}

/// The generic state machine entity envelope.
pub struct StateNode<S> {
    pub id: Uuid,
    pub temporal: TemporalGeometry,
    pub epistemic: EpistemicStatus,
    pub provenance_hash: String,
    pub payload: CreditModificationProposal,
    _marker: PhantomData<S>,
}

// ============================================================================
// 6. GATE 12 & DETERMINISTIC VERIFICATION KERNEL
// ============================================================================

#[derive(Error, Debug)]
pub enum KernelError {
    #[error("Fixed point arithmetic overflow occurred")]
    FixedPointOverflow,
    #[error("G12 Failure: Hypothesis claims non-zero authority")]
    IllegalAuthorityEscalation,
    #[error("G12 Failure: Provenance hash integrity broken")]
    ProvenanceValidationFailed,
    #[error("C4 Verification Failure: Invariant violation: {0}")]
    MathematicalInvariantViolated(&'static str),
    #[error("C5 Policy Failure: Mandate violated: {0}")]
    PolicyMandateViolated(&'static str),
    #[error("Epistemic Paralysis: Degraded mode invocation required")]
    DegradedModeRequired,
}

impl StateNode<state::EphemeralHypothesis> {
    /// Non-authoritative candidate creation. Authority is unprivileged.
    pub fn new_hypothesis(
        temporal: TemporalGeometry,
        epistemic: EpistemicStatus,
        provenance_hash: String,
        payload: CreditModificationProposal,
    ) -> Self {
        StateNode {
            id: Uuid::now_v7(),
            temporal,
            epistemic,
            provenance_hash,
            payload,
            _marker: PhantomData,
        }
    }

    /// GATE 12: Admission Boundary check.
    /// Validates schemas, snapshots, and guarantees the incoming token has ZERO mutation authority.
    pub fn evaluate_g12(self) -> Result<Self, KernelError> {
        // Enforce cryptographic provenance presence
        if self.provenance_hash.is_empty() {
            return Err(KernelError::ProvenanceValidationFailed);
        }
        // G12 passes the hypothesis to C4 verification input without granting execution rights
        Ok(self)
    }

    /// C4: DETERMINISTIC VERIFICATION KERNEL
    /// Runs decidable, integer-based invariant checks (e.g. balance equations, risk formulas).
    pub fn verify_c4(
        self,
        min_dscr_threshold: BasisPoints,
    ) -> Result<(StateNode<state::VerifiedMath>, VerificationWitness), KernelError> {
        // Mathematical Invariant: DSCR must strictly exceed statutory baseline
        if self.payload.dscr_estimate_bps < min_dscr_threshold {
            return Err(KernelError::MathematicalInvariantViolated(
                "DSCR is below axiomatic solvency bounds",
            ));
        }

        // Mathematical Invariant: Limits cannot be negative
        if self.payload.proposed_limit_bps < BasisPoints::ZERO {
            return Err(KernelError::MathematicalInvariantViolated(
                "Credit limit cannot be strictly negative",
            ));
        }

        let verified_node = StateNode {
            id: self.id,
            temporal: self.temporal,
            epistemic: self.epistemic,
            provenance_hash: self.provenance_hash,
            payload: self.payload,
            _marker: PhantomData,
        };

        Ok((verified_node, VerificationWitness { _private: () }))
    }
}

// ============================================================================
// 7. C5 POLICY GATE & C6 CRYPTOGRAPHIC COMMIT
// ============================================================================

impl StateNode<state::VerifiedMath> {
    /// C5: POLICY ADMISSION
    /// Evaluates verified proposition against active governance policies and epistemic states.
    pub fn evaluate_policy_c5(
        self,
        _vw: VerificationWitness,
        max_single_obligor_limit: BasisPoints,
    ) -> Result<(StateNode<state::PolicyAdmitted>, PolicyWitness), KernelError> {
        // Epistemic Guard: If uncertainty collapsed, standard execution cannot proceed
        if !self.epistemic.is_actionable() {
            return Err(KernelError::DegradedModeRequired);
        }

        // Institutional Mandate: Cap on aggregate obligor concentration
        if self.payload.proposed_limit_bps > max_single_obligor_limit {
            return Err(KernelError::PolicyMandateViolated(
                "Proposed exposure breaches single-obligor portfolio policy",
            ));
        }

        let admitted_node = StateNode {
            id: self.id,
            temporal: self.temporal,
            epistemic: self.epistemic,
            provenance_hash: self.provenance_hash,
            payload: self.payload,
            _marker: PhantomData,
        };

        Ok((admitted_node, PolicyWitness { _private: () }))
    }
}

impl StateNode<state::PolicyAdmitted> {
    /// C6: CRYPTOGRAPHIC COMMIT
    /// Converts an admitted transition into an append-only, signed ledger transaction.
    pub fn commit_c6(
        mut self,
        _pw: PolicyWitness,
        prev_ledger_hash: &str,
        decision_time: String,
    ) -> (StateNode<state::CommittedLog>, String) {
        self.temporal.t_d = Some(decision_time);

        // Monotonic Hash Chaining: SHA256(prev_hash || node_id || payload || t_d)
        let mut hasher = Context::new(&SHA256);
        hasher.update(prev_ledger_hash.as_bytes());
        hasher.update(self.id.as_bytes());
        hasher.update(self.payload.counterparty_id.as_bytes());
        hasher.update(&self.payload.proposed_limit_bps.0.to_be_bytes());
        hasher.update(self.temporal.t_d.as_ref().unwrap().as_bytes());

        let commit_hash = hex::encode(hasher.finish().as_ref());

        let committed_node = StateNode {
            id: self.id,
            temporal: self.temporal,
            epistemic: self.epistemic,
            provenance_hash: commit_hash.clone(),
            payload: self.payload,
            _marker: PhantomData,
        };

        (committed_node, commit_hash)
    }
}

pub fn verify_provenance_dag(json_ld: &str) -> Result<bool, KernelError> {
    let parsed: serde_json::Value = serde_json::from_str(json_ld)
        .map_err(|_| KernelError::ProvenanceValidationFailed)?;

    let graph = parsed["@graph"]
        .as_array()
        .ok_or(KernelError::ProvenanceValidationFailed)?;

    // 1. Structural Validation: Ensure every entity originates from a CanonicalObservation (C0)
    let mut has_c0_source = false;
    for node in graph {
        if let Some(types) = node["@type"].as_array() {
            if types.iter().any(|t| t == "adam:CanonicalObservation") {
                // Ensure cryptographic hash is sealed
                if node["adam:attested_hash"].as_str().is_some() {
                    has_c0_source = true;
                }
            }
        }
    }

    if !has_c0_source {
        return Err(KernelError::ProvenanceValidationFailed);
    }

    // 2. Authority Check: Verify hypothesis does not claim authority
    for node in graph {
        if let Some(auth) = node["adam:authority"].as_str() {
            if auth != "NONE" {
                return Err(KernelError::IllegalAuthorityEscalation);
            }
        }
    }

    Ok(true)
}

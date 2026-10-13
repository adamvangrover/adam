use crate::memory::canonical::compute_canonical_hash;
use crate::memory::provenance::MemoryProvenance;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum MemoryTier {
    M0, // Working memory (ephemeral, untrusted)
    M1, // Episodic memory (historical evidence)
    M2, // Semantic memory (validated facts, versioned & evidence-linked)
    M3, // Governed state (admitted canonical facts, state machine controlled)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryKind {
    Observation,
    Hypothesis,
    ValidatedClaim,
    Decision,
    ExecutionTrace,
    Policy,
    RepositoryArtifact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicStatus {
    UNVERIFIED,
    PENDING,
    VERIFIED,
    CONTESTED,
    SUPERSEDED,
    REJECTED,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    pub memory_id: String,
    pub schema_version: String,
    pub tier: MemoryTier,
    pub kind: MemoryKind,
    pub epistemic_status: EpistemicStatus,
    pub content: Value,
    pub content_hash: String,
    #[serde(default)]
    pub source_refs: Vec<String>,
    pub provenance: MemoryProvenance,
    pub correlation_id: String,
    pub effective_at: Option<String>,
    pub observed_at: Option<String>,
    pub supersedes: Option<String>,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    #[error("Schema version mismatch. Expected 1.0, got {0}")]
    InvalidSchemaVersion(String),

    #[error("Content hash mismatch. Expected {expected}, got {actual}")]
    ContentHashMismatch { expected: String, actual: String },

    #[error("Invalid provenance: missing activity_id, agent_id, or policy_version")]
    InvalidProvenance,

    #[error("Unauthorized M3 promotion: M3 tier requires VERIFIED epistemic status and authorized policy admission")]
    UnauthorizedM3Promotion,

    #[error("M0 tier working memory cannot hold VERIFIED status directly")]
    UnverifiedM0Claim,

    #[error("M2 semantic memory requires non-empty source_refs evidence linking")]
    MissingSourceRefsForM2,

    #[error("Duplicate memory write for memory_id: {0} or content_hash: {1}")]
    DuplicateMemoryWrite(String, String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionReceipt {
    pub memory_id: String,
    pub tier: MemoryTier,
    pub epistemic_status: EpistemicStatus,
    pub canonical_hash: String,
    pub correlation_id: String,
    pub timestamp_rfc3339: String,
}

#[derive(Debug, Default)]
pub struct AdmissionEngine {
    nodes: HashMap<String, MemoryNode>,
    hashes: HashSet<String>,
}

impl AdmissionEngine {
    pub fn new() -> Self {
        AdmissionEngine {
            nodes: HashMap::new(),
            hashes: HashSet::new(),
        }
    }

    /// Evaluates candidate memory node according to the 4-tier memory governance rules.
    pub fn admit_node(&mut self, candidate: MemoryNode) -> Result<AdmissionReceipt, AdmissionError> {
        // 1. Schema version check
        if candidate.schema_version != "1.0" {
            return Err(AdmissionError::InvalidSchemaVersion(candidate.schema_version.clone()));
        }

        // 2. Provenance completeness check
        if !candidate.provenance.is_valid() {
            return Err(AdmissionError::InvalidProvenance);
        }

        // 3. Content hash verification
        let computed_hash = compute_canonical_hash(&candidate.content);
        if candidate.content_hash.to_lowercase() != computed_hash.to_lowercase() {
            return Err(AdmissionError::ContentHashMismatch {
                expected: computed_hash,
                actual: candidate.content_hash.clone(),
            });
        }

        // 4. Duplicate write check
        if self.nodes.contains_key(&candidate.memory_id) || self.hashes.contains(&candidate.content_hash) {
            return Err(AdmissionError::DuplicateMemoryWrite(
                candidate.memory_id.clone(),
                candidate.content_hash.clone(),
            ));
        }

        // 5. Tier & Epistemic Status Boundary Checks
        match candidate.tier {
            MemoryTier::M0 => {
                // M0 cannot claim VERIFIED status directly
                if candidate.epistemic_status == EpistemicStatus::VERIFIED {
                    return Err(AdmissionError::UnverifiedM0Claim);
                }
            }
            MemoryTier::M1 => {
                // Episodic memory allowed
            }
            MemoryTier::M2 => {
                // M2 semantic memory must have source refs (evidence linking)
                if candidate.source_refs.is_empty() {
                    return Err(AdmissionError::MissingSourceRefsForM2);
                }
            }
            MemoryTier::M3 => {
                // M3 Governed state requires VERIFIED epistemic status
                if candidate.epistemic_status != EpistemicStatus::VERIFIED {
                    return Err(AdmissionError::UnauthorizedM3Promotion);
                }
            }
        }

        let admission_timestamp = candidate
            .observed_at
            .clone()
            .or_else(|| candidate.effective_at.clone())
            .unwrap_or_else(|| "2026-01-01T00:00:00Z".to_string());

        let receipt = AdmissionReceipt {
            memory_id: candidate.memory_id.clone(),
            tier: candidate.tier,
            epistemic_status: candidate.epistemic_status,
            canonical_hash: candidate.content_hash.clone(),
            correlation_id: candidate.correlation_id.clone(),
            timestamp_rfc3339: admission_timestamp,
        };

        self.hashes.insert(candidate.content_hash.clone());
        self.nodes.insert(candidate.memory_id.clone(), candidate);

        Ok(receipt)
    }

    pub fn get_node(&self, memory_id: &str) -> Option<&MemoryNode> {
        self.nodes.get(memory_id)
    }

    pub fn list_nodes(&self) -> Vec<&MemoryNode> {
        self.nodes.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_candidate() -> MemoryNode {
        let content = json!({"fact": "credit spread widened"});
        let hash = compute_canonical_hash(&content);
        MemoryNode {
            memory_id: "mem-01".to_string(),
            schema_version: "1.0".to_string(),
            tier: MemoryTier::M1,
            kind: MemoryKind::Observation,
            epistemic_status: EpistemicStatus::UNVERIFIED,
            content,
            content_hash: hash,
            source_refs: vec!["ref1".to_string()],
            provenance: MemoryProvenance {
                activity_id: "act-1".to_string(),
                agent_id: "agent-1".to_string(),
                policy_version: "1.0".to_string(),
                parent_record_ids: vec![],
            },
            correlation_id: "corr-1".to_string(),
            effective_at: None,
            observed_at: None,
            supersedes: None,
        }
    }

    #[test]
    fn test_valid_m1_admission() {
        let mut engine = AdmissionEngine::new();
        let cand = valid_candidate();
        let res = engine.admit_node(cand);
        assert!(res.is_ok());
    }

    #[test]
    fn test_reject_m3_unverified() {
        let mut engine = AdmissionEngine::new();
        let mut cand = valid_candidate();
        cand.tier = MemoryTier::M3;
        cand.epistemic_status = EpistemicStatus::UNVERIFIED;
        let res = engine.admit_node(cand);
        assert_eq!(res.unwrap_err(), AdmissionError::UnauthorizedM3Promotion);
    }
}

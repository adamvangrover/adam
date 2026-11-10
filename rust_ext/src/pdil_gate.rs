use pyo3::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[pyclass(frozen)]
pub struct RustPDILGatekeeper {
    #[pyo3(get)]
    pub min_conviction: f64,
    #[pyo3(get)]
    pub max_vpin: f64,
    #[pyo3(get)]
    pub min_dscr: f64,
}

#[pymethods]
impl RustPDILGatekeeper {
    #[new]
    #[pyo3(signature = (min_conviction=0.85, max_vpin=0.82, min_dscr=1.05))]
    pub fn new(min_conviction: f64, max_vpin: f64, min_dscr: f64) -> Self {
        Self {
            min_conviction,
            max_vpin,
            min_dscr,
        }
    }

    /// Evaluates Layer 0 x Layer 1 in Rust without holding the Python GIL.
    /// Returns (layer_0_gate: u8, final_score: f64, canonical_sha256: String, status: String)
    pub fn evaluate_gate(
        &self,
        py: Python<'_>,
        raw_json_bytes: &[u8],
        jev_conviction: f64,
        vpin_toxicity: f64,
        stressed_dscr: f64,
        cedar_forbid_triggered: bool,
        raw_quality_score: f64,
    ) -> PyResult<(u8, f64, String, String)> {
        py.allow_threads(|| {
            // 1. JCS / Canonical JSON Admission & W3C PROV-O SHA-256
            let parsed: Value = serde_json::from_slice(raw_json_bytes)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JCS_REJECT: {e}")))?;
            let canonical_str = serde_json::to_string(&parsed)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JCS_CANON_ERR: {e}")))?;
            let mut hasher = Sha256::new();
            hasher.update(canonical_str.as_bytes());
            let sha256_hex = format!("{:x}", hasher.finalize());

            // 2. Multiplicative Layer 0 Hard Safety Tripwires (No Reward Over Safety)
            if cedar_forbid_triggered {
                return Ok((0, 0.0, sha256_hex, "BLOCK_CEDAR_FORBID".to_string()));
            }
            if vpin_toxicity > self.max_vpin {
                return Ok((0, 0.0, sha256_hex, "BLOCK_CBF_VPIN_TOXICITY".to_string()));
            }
            if stressed_dscr < self.min_dscr {
                return Ok((0, 0.0, sha256_hex, "BLOCK_DSCR_FLOOR_BREACH".to_string()));
            }
            if jev_conviction < self.min_conviction {
                return Ok((0, 0.0, sha256_hex, "ABSTAIN_JEV_LOW_CONVICTION".to_string()));
            }

            // 3. Lexicographic Product: Layer 0 (1) * Layer 1 Quality Score
            let final_score = 1.0 * raw_quality_score.clamp(0.0, 5.0);
            let route = if final_score >= 4.0 {
                "COMMIT_CANDIDATE"
            } else if final_score >= 3.0 {
                "HUMAN_ESCALATE"
            } else {
                "BLOCK_AND_LOG"
            };
            Ok((1, final_score, sha256_hex, route.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gatekeeper_blocks_cedar() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let gate = RustPDILGatekeeper::new(0.85, 0.82, 1.05);
            let payload = b"{\"key\":\"val\"}";
            let (l0, score, _sha, status) = gate
                .evaluate_gate(py, payload, 0.90, 0.10, 1.50, true, 4.5)
                .unwrap();
            assert_eq!(l0, 0);
            assert_eq!(score, 0.0);
            assert_eq!(status, "BLOCK_CEDAR_FORBID");
        });
    }

    #[test]
    fn test_gatekeeper_passes_valid() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let gate = RustPDILGatekeeper::new(0.85, 0.82, 1.05);
            let payload = b"{\"key\":\"val\"}";
            let (l0, score, sha, status) = gate
                .evaluate_gate(py, payload, 0.90, 0.10, 1.50, false, 4.5)
                .unwrap();
            assert_eq!(l0, 1);
            assert_eq!(score, 4.5);
            assert!(!sha.is_empty());
            assert_eq!(status, "COMMIT_CANDIDATE");
        });
    }
}

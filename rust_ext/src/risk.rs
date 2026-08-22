use pyo3::prelude::*;
use crate::error::KernelError;

#[pyclass]
#[derive(Clone)]
pub struct RiskResult {
    #[pyo3(get)]
    pub value: f64,
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub confidence: f64,
    #[pyo3(get)]
    pub kernel_version: String,
    #[pyo3(get)]
    pub deterministic: bool,
}

#[pyfunction]
pub fn calculate_var(py: Python<'_>, returns: Vec<f64>, confidence: f64, method: String) -> PyResult<RiskResult> {
    py.allow_threads(move || {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(KernelError::InputError("Confidence must be between 0 and 1".to_string()).into());
        }

        if returns.is_empty() {
            return Err(KernelError::InputError("Returns array cannot be empty".to_string()).into());
        }

        let mut sorted_returns = returns.clone();
        // Deterministic sort for floating points, avoiding NaN panic
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((1.0 - confidence) * (sorted_returns.len() as f64)).floor() as usize;
        // Prevent out of bounds
        let index = index.min(sorted_returns.len() - 1);

        let var_value = -sorted_returns[index]; // VaR is typically expressed as a positive loss value

        Ok(RiskResult {
            value: var_value,
            method: method,
            confidence: confidence,
            kernel_version: "risk-kernel-1".to_string(),
            deterministic: true,
        })
    })
}

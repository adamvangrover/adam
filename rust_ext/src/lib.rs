use pyo3::prelude::*;

mod error;
pub mod statistics;
mod risk;
pub mod portfolio;
pub mod stress;

/// Deterministic numerical math kernel for AFOS
#[pymodule]
fn rust_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<risk::RiskResult>()?;
    m.add_function(wrap_pyfunction!(risk::calculate_var, m)?)?;
    Ok(())
}

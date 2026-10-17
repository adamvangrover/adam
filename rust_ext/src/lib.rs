use pyo3::prelude::*;

pub mod error;
pub mod portfolio;
pub mod risk;
pub mod statistics;
pub mod stress;

/// Deterministic numerical math kernel for AFOS
#[pymodule]
fn rust_ext(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<risk::RiskResult>()?;
    m.add_function(wrap_pyfunction!(risk::calculate_var, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::calculate_mean, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::calculate_variance, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio::aggregate_exposure, m)?)?;
    m.add_function(wrap_pyfunction!(stress::apply_stress_scenario, m)?)?;
    Ok(())
}

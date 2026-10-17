use pyo3::prelude::*;

#[pyfunction]
pub fn apply_stress_scenario(py: Python<'_>, positions: Vec<f64>, stress_factor: f64) -> PyResult<Vec<f64>> {
    py.allow_threads(move || {
        Ok(positions.iter().map(|&x| x * (1.0 - stress_factor)).collect())
    })
}

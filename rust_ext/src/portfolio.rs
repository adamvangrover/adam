use pyo3::prelude::*;

#[pyfunction]
pub fn aggregate_exposure(py: Python<'_>, positions: Vec<f64>) -> PyResult<f64> {
    py.allow_threads(move || {
        Ok(positions.iter().sum())
    })
}

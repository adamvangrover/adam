use pyo3::prelude::*;

#[pyfunction]
pub fn calculate_mean(py: Python<'_>, values: Vec<f64>) -> PyResult<f64> {
    py.allow_threads(move || {
        if values.is_empty() {
            return Ok(0.0);
        }
        let sum: f64 = values.iter().sum();
        Ok(sum / (values.len() as f64))
    })
}

#[pyfunction]
pub fn calculate_variance(py: Python<'_>, values: Vec<f64>) -> PyResult<f64> {
    py.allow_threads(move || {
        if values.is_empty() {
            return Ok(0.0);
        }
        let mean = values.iter().sum::<f64>() / (values.len() as f64);
        let variance: f64 = values.iter().map(|value| {
            let diff = mean - *value;
            diff * diff
        }).sum();
        Ok(variance / (values.len() as f64))
    })
}

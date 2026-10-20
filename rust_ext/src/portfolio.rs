use pyo3::prelude::*;

#[pyfunction]
pub fn aggregate_exposure(py: Python<'_>, positions: Vec<f64>) -> PyResult<f64> {
    py.allow_threads(move || {
        Ok(positions.iter().sum())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_exposure() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions = vec![100.0, 250.5, 49.5];
            let total = aggregate_exposure(py, positions).unwrap();
            assert_eq!(total, 400.0);
        });
    }
}

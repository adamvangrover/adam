use pyo3::prelude::*;

#[pyfunction]
pub fn apply_stress_scenario(py: Python<'_>, positions: Vec<f64>, stress_factor: f64) -> PyResult<Vec<f64>> {
    py.allow_threads(move || {
        Ok(positions.iter().map(|&x| x * (1.0 - stress_factor)).collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_stress_scenario() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let positions = vec![100.0, 200.0];
            let stressed = apply_stress_scenario(py, positions, 0.20).unwrap();
            assert_eq!(stressed, vec![80.0, 160.0]);
        });
    }
}

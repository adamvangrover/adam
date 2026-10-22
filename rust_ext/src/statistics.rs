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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_mean() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let values = vec![10.0, 20.0, 30.0];
            let mean = calculate_mean(py, values).unwrap();
            assert_eq!(mean, 20.0);
        });
    }

    #[test]
    fn test_calculate_variance() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
            let var = calculate_variance(py, values).unwrap();
            assert!((var - 4.0).abs() < 1e-5);
        });
    }
}

use std::f64;

pub fn calculate_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f64 = values.iter().sum();
    sum / (values.len() as f64)
}

pub fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = calculate_mean(values);
    let variance: f64 = values.iter().map(|value| {
        let diff = mean - *value;
        diff * diff
    }).sum();
    variance / (values.len() as f64)
}

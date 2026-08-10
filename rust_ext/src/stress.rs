pub fn apply_stress_scenario(positions: &[f64], stress_factor: f64) -> Vec<f64> {
    positions.iter().map(|&x| x * (1.0 - stress_factor)).collect()
}

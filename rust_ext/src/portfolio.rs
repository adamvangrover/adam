pub fn aggregate_exposure(positions: &[f64]) -> f64 {
    positions.iter().sum()
}

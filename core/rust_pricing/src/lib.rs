use pyo3::prelude::*;

mod pricing;
use pricing::{MarketParams, calculate_quotes};

#[pyfunction]
fn get_quotes(mid_price: f64, volatility: f64, inventory: f64, risk_aversion: f64, time_horizon: f64, liquidity_param: f64) -> PyResult<(f64, f64)> {
    let params = MarketParams {
        mid_price,
        volatility,
        inventory,
        risk_aversion,
        time_horizon,
        liquidity_param,
    };
    Ok(calculate_quotes(params))
}

#[pymodule]
fn rust_pricing(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_quotes, m)?)?;
    Ok(())
}

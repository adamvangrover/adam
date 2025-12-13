// core/rust_pricing/src/pricing.rs

pub struct MarketParams {
    pub mid_price: f64,
    pub volatility: f64,
    pub inventory: f64,
    pub risk_aversion: f64, // Gamma
    pub time_horizon: f64,  // T - t
    pub liquidity_param: f64, // Kappa
}

pub fn calculate_quotes(params: MarketParams) -> (f64, f64) {
    // 1. Calculate Reservation Price (r)
    // r = s - q * gamma * sigma^2 * (T - t)
    let reservation_price = params.mid_price - (params.inventory * params.risk_aversion * params.volatility.powi(2) * params.time_horizon);

    // 2. Calculate Optimal Half-Spread (delta)
    // delta = (1/gamma) * ln(1 + gamma/kappa) + 0.5 * gamma * sigma^2 * (T-t)
    let spread_term_1 = (1.0 / params.risk_aversion) * (1.0 + params.risk_aversion / params.liquidity_param).ln();
    let spread_term_2 = 0.5 * params.risk_aversion * params.volatility.powi(2) * params.time_horizon;
    let half_spread = spread_term_1 + spread_term_2;

    // 3. Generate Quotes
    let bid_price = reservation_price - half_spread;
    let ask_price = reservation_price + half_spread;

    (bid_price, ask_price)
}

use std::sync::atomic::{AtomicU64, Ordering};

// 3.1 The Avellaneda-Stoikov Framework

pub struct AvellanedaStoikovStrategy {
    pub gamma: AtomicU64, // Risk aversion (scaled)
    pub kappa: AtomicU64, // Arrival intensity (scaled)
    pub sigma: AtomicU64, // Volatility
    pub inventory: AtomicU64,
}

impl AvellanedaStoikovStrategy {
    pub fn new(gamma: f64, kappa: f64, sigma: f64) -> Self {
        Self {
            gamma: AtomicU64::new((gamma * 1_000_000.0) as u64),
            kappa: AtomicU64::new((kappa * 1_000_000.0) as u64),
            sigma: AtomicU64::new((sigma * 1_000_000.0) as u64),
            inventory: AtomicU64::new(0),
        }
    }

    pub fn calculate_reservation_price(&self, mid_price: f64, time_remaining: f64) -> f64 {
        let s = mid_price;
        let q = self.inventory.load(Ordering::Relaxed) as f64; // Simplified: inventory is signed in reality
        let gamma = self.gamma.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let sigma = self.sigma.load(Ordering::Relaxed) as f64 / 1_000_000.0;

        // r(s, t) = s(t) - q(t) * gamma * sigma^2 * (T - t)
        s - (q * gamma * sigma.powi(2) * time_remaining)
    }

    pub fn update_parameters(&self, new_gamma: f64, new_kappa: f64) {
        self.gamma.store((new_gamma * 1_000_000.0) as u64, Ordering::Relaxed);
        self.kappa.store((new_kappa * 1_000_000.0) as u64, Ordering::Relaxed);
    }
}

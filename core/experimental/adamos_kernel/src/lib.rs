use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

/// Represents an Agent in the WASM sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub capabilities: Vec<String>,
    pub status: AgentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Thinking,
    Sleeping,
    Dead,
}

/// A message passed between agents via the Kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub sender: String,
    pub recipient: String,
    pub content: String,
    pub timestamp: u64,
}

/// The AdamOS Kernel (Orchestrator)
#[wasm_bindgen]
pub struct Kernel {
    // Wrapped in Arc/Mutex for thread safety, though WASM is single-threaded usually
    agents: Arc<Mutex<HashMap<String, Agent>>>,
    message_bus: Arc<Mutex<Vec<Message>>>,
}

#[wasm_bindgen]
impl Kernel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Kernel {
            agents: Arc::new(Mutex::new(HashMap::new())),
            message_bus: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Registers a new agent (e.g., loaded from WASM)
    #[wasm_bindgen(js_name = registerAgent)]
    pub fn register_agent(&self, name: String, capabilities: Vec<String>) -> String {
        let id = Uuid::new_v4().to_string();
        // Simple mapping for capabilities from Vec<String> to internal logic if needed
        let agent = Agent {
            id: id.clone(),
            name: name.clone(),
            capabilities,
            status: AgentStatus::Idle,
        };

        let mut agents = self.agents.lock().unwrap();
        agents.insert(id.clone(), agent);
        // console_log!("Kernel: Registered Agent {} ({})", name, id);
        id
    }

    /// Routes a message
    #[wasm_bindgen(js_name = sendMessage)]
    pub fn send_message(&self, sender_id: &str, recipient_id: &str, content: String) {
        let msg = Message {
            id: Uuid::new_v4().to_string(),
            sender: sender_id.to_string(),
            recipient: recipient_id.to_string(),
            content,
            timestamp: 0, // Mock timestamp
        };

        let mut bus = self.message_bus.lock().unwrap();
        bus.push(msg);
    }

    /// Simulates a kernel tick (routing, health checks)
    pub fn tick(&self) -> usize {
        let agents = self.agents.lock().unwrap();
        agents.len()
    }
}

// --- WASM EXPORTS FOR PHYSICS ENGINE ---

/// Cumulative Distribution Function for Standard Normal Distribution
fn norm_cdf(x: f64) -> f64 {
    // Approximation of Error Function (erf)
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / 2.0_f64.sqrt();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

/// Black-Scholes Option Pricing (Call)
/// S: Current Stock Price
/// K: Strike Price
/// T: Time to Maturity (in years)
/// r: Risk-free Interest Rate
/// sigma: Volatility
#[wasm_bindgen(js_name = calculateOptionPrice)]
pub fn calculate_option_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
    if t <= 0.0 {
        return if is_call { (s - k).max(0.0) } else { (k - s).max(0.0) };
    }

    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    if is_call {
        s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
    } else {
        k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
    }
}
